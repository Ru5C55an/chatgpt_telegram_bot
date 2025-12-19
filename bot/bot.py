import io
import logging
import asyncio
import traceback
import html
import json
import re
from datetime import datetime, timedelta
import openai

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    LabeledPrice
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils
import constants as C

import base64

# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}
user_wait_messages = {}
user_media_groups = {}

def get_localized_text(key, user_id):
    language = db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE
    return config.locales[language].get(key, config.locales["en"][key])

def get_formatted_date(date, user_id):
    if not date:
        return ""
    language = db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE
    if language == "ru":
        return date.strftime("%d.%m.%Y %H:%M")
    return date.strftime("%Y-%m-%d %H:%M")

def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, C.DB_CURRENT_DIALOG_ID) is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, C.DB_CURRENT_MODEL) is None:
        db.set_user_attribute(user.id, C.DB_CURRENT_MODEL, C.DEFAULT_MODEL)

    if db.get_user_attribute(user.id, C.DB_CURRENT_LANGUAGE) is None:
        db.set_user_attribute(user.id, C.DB_CURRENT_LANGUAGE, C.DEFAULT_LANGUAGE)

    if db.get_user_attribute(user.id, C.DB_USER_PROFILE) is None:
        db.set_user_attribute(user.id, C.DB_USER_PROFILE, {
            C.PROFILE_HEIGHT: None,
            C.PROFILE_WEIGHT: None,
            C.PROFILE_FITNESS_LEVEL: None,
            C.PROFILE_GOALS: None,
            C.PROFILE_GENDER: None,
        })

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, C.DB_N_USED_TOKENS)
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                C.DB_N_INPUT_TOKENS: 0,
                C.DB_N_OUTPUT_TOKENS: n_used_tokens
            }
        }
        db.set_user_attribute(user.id, C.DB_N_USED_TOKENS, new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, C.DB_N_TRANSCRIBED_SECONDS) is None:
        db.set_user_attribute(user.id, C.DB_N_TRANSCRIBED_SECONDS, 0.0)

    # image generation
    if db.get_user_attribute(user.id, C.DB_N_GENERATED_IMAGES) is None:
        db.set_user_attribute(user.id, C.DB_N_GENERATED_IMAGES, 0)


async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         text = message.text or message.caption
         if text is not None and ("@" + context.bot.username) in text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except:
         return True
     else:
         return False


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())
    db.start_new_dialog(user_id)

    reply_text = get_localized_text(C.LOC_WELCOME, user_id)
    reply_text += get_localized_text(C.LOC_HELP, user_id)

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_profile_handle(update, context)


async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())
    await update.message.reply_text(get_localized_text(C.LOC_HELP, user_id), parse_mode=ParseMode.HTML)


async def help_group_chat_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update, context, update.message.from_user)
     user_id = update.message.from_user.id
     db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

     text = get_localized_text(C.LOC_HELP_GROUP_CHAT, user_id).format(bot_username="@" + context.bot.username)

     await update.message.reply_text(text, parse_mode=ParseMode.HTML)
     await update.message.reply_video(config.help_group_chat_video_path)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text(get_localized_text(C.LOC_NO_MESSAGE_TO_RETRY, user_id))
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"])

async def _vision_message_handle_fn(
    update: Update, context: CallbackContext, message=None, image_buffers: list = None
):
    logger.info('_vision_message_handle_fn')
    user_id = update.message.from_user.id
    current_model = db.get_user_attribute(user_id, C.DB_CURRENT_MODEL)
    if current_model not in config.models["available_text_models"]:
        current_model = "gpt-5-mini-2025-08-07"
        db.set_user_attribute(user_id, C.DB_CURRENT_MODEL, current_model)

    if current_model not in ["gpt-4-vision-preview", "gpt-4o", "gpt-5-mini-2025-08-07"]:
        await update.message.reply_text(
            get_localized_text(C.LOC_IMAGES_PROCESSING_NOT_AVAILABLE, user_id),
            parse_mode=ParseMode.HTML,
        )
        return

    chat_mode = db.get_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE)
    if chat_mode not in config.chat_modes.keys():
        chat_mode = "ai_trainer"
        db.set_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE, chat_mode)

    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    bufs = []
    if image_buffers is not None:
        bufs = image_buffers
    elif update.message.photo:
        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)

        # store file in memory, not on disk
        buf = io.BytesIO()
        await photo_file.download_to_memory(buf)
        buf.name = "image.jpg"  # file extension is required
        buf.seek(0)  # move cursor to the beginning of the buffer
        bufs.append(buf)

    # in case of CancelledError
    n_input_tokens, n_output_tokens = 0, 0

    try:
        # send placeholder message to user
        placeholder_message = await update.message.reply_text("...")
        # Use the message parameter if provided (e.g., from voice transcription), otherwise get from update
        _message = message if message is not None else (update.message.caption or update.message.text or '')

        # send typing action
        await update.message.chat.send_action(action="typing")

        dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
        parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
            config.chat_modes[chat_mode]["parse_mode"]
        ]

        chatgpt_instance = openai_utils.ChatGPT(model=current_model)
        user_profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
        if config.enable_message_streaming:
            gen = chatgpt_instance.send_vision_message_stream(
                _message,
                dialog_messages=dialog_messages,
                image_buffers=bufs,
                chat_mode=chat_mode,
                user_language=db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE,
                user_profile=user_profile,
            )
        else:
            (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = await chatgpt_instance.send_vision_message(
                _message,
                dialog_messages=dialog_messages,
                image_buffers=bufs,
                chat_mode=chat_mode,
                user_language=db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE,
                user_profile=user_profile,
            )

            async def fake_gen():
                yield "finished", answer, (
                    n_input_tokens,
                    n_output_tokens,
                ), n_first_dialog_messages_removed

            gen = fake_gen()

        prev_answer = ""
        async for gen_item in gen:
            (
                status,
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = gen_item

            answer = answer[:4096]  # telegram message limit

            # update only when 100 new symbols are ready
            if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                continue

            try:
                await context.bot.edit_message_text(
                    answer,
                    chat_id=placeholder_message.chat_id,
                    message_id=placeholder_message.message_id,
                    parse_mode=parse_mode,
                )
            except telegram.error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    continue
                else:
                    await context.bot.edit_message_text(
                        answer,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                    )

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            prev_answer = answer

        # update user data
        if bufs:
            # For simplicity, we only store the first image in DB for history display if needed
            # or we could store all, but the DB schema might need change.
            # Storing simplified message for history.
            base_images = [base64.b64encode(b.getvalue()).decode("utf-8") for b in bufs]
            new_dialog_message = {"user": [
                        {
                            "type": "text",
                            "text": _message,
                        }
                    ] + [
                        {
                            "type": "image",
                            "image": img,
                        } for img in base_images
                    ]
                , "bot": answer, "date": datetime.now()}
        else:
            new_dialog_message = {"user": [{"type": "text", "text": _message}], "bot": answer, "date": datetime.now()}
        
        db.set_dialog_messages(
            user_id,
            db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
            dialog_id=None
        )

        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

    except openai.error.RateLimitError:
        error_text = get_localized_text(C.LOC_OPENAI_RATE_LIMIT_ERROR, user_id)
        logger.error(f"Rate limit reached for user {user_id}")
        await update.message.reply_text(error_text, parse_mode=ParseMode.HTML)
        return

    except asyncio.CancelledError:
        # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
        raise

    except Exception as e:
        if "Timed out" in str(e) and prev_answer:
            logger.warning(f"Timeout in _vision_message_handle_fn, but response was partially sent. Error: {e}")
            return
        error_text = get_localized_text(C.LOC_ERROR_MESSAGE, user_id).format(reason=str(e))
        logger.error(error_text)
        await update.message.reply_text(error_text)
        return

async def unsupport_message_handle(update: Update, context: CallbackContext, message=None):
    user_id = update.message.from_user.id
    error_text = get_localized_text(C.LOC_UNSUPPORTED_MESSAGE_TYPE, user_id)
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return

async def message_handle(update: Update, context: CallbackContext, message=None, **kwargs):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text or update.message.caption or ""

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    
    # handle media groups (multiple photos)
    if update.message.media_group_id and not kwargs.get("image_buffers"):
        mg_id = update.message.media_group_id
        if mg_id not in user_media_groups:
            user_media_groups[mg_id] = {"images": [], "captions": [], "task": None}
        
        # collect photo
        if update.message.photo:
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            buf = io.BytesIO()
            await photo_file.download_to_memory(buf)
            buf.name = "image.jpg"
            buf.seek(0)
            user_media_groups[mg_id]["images"].append(buf)
        
        # collect caption
        caption = update.message.caption or update.message.text
        if caption:
            user_media_groups[mg_id]["captions"].append(caption)
            
        if user_media_groups[mg_id]["task"] is None:
            async def handle_media_group():
                await asyncio.sleep(0.8) # wait for other images to arrive
                data = user_media_groups.get(mg_id)
                if not data: return
                # Use unique captions in order
                seen = set()
                unique_captions = [x for x in data["captions"] if not (x in seen or seen.add(x))]
                combined_message = " ".join(unique_captions) or message or ""
                await message_handle(update, context, message=combined_message, image_buffers=data["images"])

            user_media_groups[mg_id]["task"] = asyncio.create_task(handle_media_group())
        return
    
    # Handle profile field text input
    if C.CONTEXT_PROFILE_FIELD_EDITING in context.user_data:
        field = context.user_data[C.CONTEXT_PROFILE_FIELD_EDITING]
        value = _message.strip()
        profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
        
        if field == C.PROFILE_HEIGHT or field == C.PROFILE_WEIGHT:
            try:
                numeric_value = float(value)
                profile[field] = numeric_value
                db.set_user_attribute(user_id, C.DB_USER_PROFILE, profile)
                text = get_localized_text(C.LOC_PROFILE_UPDATED, user_id)
                del context.user_data[C.CONTEXT_PROFILE_FIELD_EDITING]
                await update.message.reply_text(text, parse_mode=ParseMode.HTML)
                await prompt_next_empty_profile_field(user_id, context, update)
                return
            except ValueError:
                text = get_localized_text(C.LOC_PROFILE_INVALID_NUMBER, user_id)
                await update.message.reply_text(text, parse_mode=ParseMode.HTML)
                return
        elif field == C.PROFILE_GOALS:
            profile[field] = value
            db.set_user_attribute(user_id, C.DB_USER_PROFILE, profile)
            text = get_localized_text(C.LOC_PROFILE_UPDATED, user_id)
            del context.user_data[C.CONTEXT_PROFILE_FIELD_EDITING]
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            await prompt_next_empty_profile_field(user_id, context, update)
            return

    # Check quota for free users
    if not await check_user_quota(user_id, update, context):
        return
    
    chat_mode = db.get_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE)
    if chat_mode not in config.chat_modes.keys():
        chat_mode = C.DEFAULT_CHAT_MODE
        db.set_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE, chat_mode)

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    current_model = db.get_user_attribute(user_id, C.DB_CURRENT_MODEL)
    if current_model not in config.models["available_text_models"]:
        current_model = C.DEFAULT_MODEL
        db.set_user_attribute(user_id, C.DB_CURRENT_MODEL, current_model)

    async def message_handle_fn():
        db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                 await update.message.reply_text(get_localized_text(C.LOC_EMPTY_MESSAGE, user_id), parse_mode=ParseMode.HTML)
                 return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            user_profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode, user_language=db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE, user_profile=user_profile)
            else:
                answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    _message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode,
                    user_language=db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE,
                    user_profile=user_profile,
                )

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            prev_answer = ""
            
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = answer[:4096]  # telegram message limit
                    
                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding
                
                prev_answer = answer
            
            # update user data
            new_dialog_message = {"user": [{"type": "text", "text": _message}], "bot": answer, "date": datetime.now()}

            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except openai.error.RateLimitError:
            error_text = get_localized_text(C.LOC_OPENAI_RATE_LIMIT_ERROR, user_id)
            logger.error(f"Rate limit reached for user {user_id}")
            await update.message.reply_text(error_text, parse_mode=ParseMode.HTML)
            return

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            if "Timed out" in str(e) and prev_answer:
                logger.warning(f"Timeout in message_handle_fn, but response was partially sent. Error: {e}")
                return
            error_text = get_localized_text(C.LOC_ERROR_MESSAGE, user_id).format(reason=str(e))
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = get_localized_text(C.LOC_CONTEXT_REMOVED_FIRST, user_id)
            else:
                text = get_localized_text(C.LOC_CONTEXT_REMOVED_MULTIPLE, user_id).format(n_messages=n_first_dialog_messages_removed)
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        # get image_buffers from kwargs or collected from media group
        image_buffers = kwargs.get("image_buffers")

        if current_model in ["gpt-4-vision-preview", "gpt-4o", "gpt-5-mini-2025-08-07"] or (update.message.photo is not None and len(update.message.photo) > 0) or image_buffers:

            if current_model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-5-mini-2025-08-07"]:
                current_model = "gpt-5-mini-2025-08-07"
                db.set_user_attribute(user_id, C.DB_CURRENT_MODEL, C.OPENAI_MODEL_GPT_5_MINI)
            task = asyncio.create_task(
                _vision_message_handle_fn(update, context, message=_message, image_buffers=image_buffers)
            )
        else:
            task = asyncio.create_task(
                message_handle_fn()
            )            

        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text(get_localized_text(C.LOC_CANCELED, user_id), parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]
            
            # delete "wait for reply" messages
            if user_id in user_wait_messages:
                for message_id in user_wait_messages[user_id]:
                    try:
                        await context.bot.delete_message(chat_id=update.message.chat_id, message_id=message_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete 'wait for reply' message: {e}")
                user_wait_messages[user_id] = []
            
            # cleanup media groups if any
            if update.message and update.message.media_group_id in user_media_groups:
                del user_media_groups[update.message.media_group_id]


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = get_localized_text(C.LOC_WAIT_FOR_REPLY, user_id)
        wait_message = await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        
        if user_id not in user_wait_messages:
            user_wait_messages[user_id] = []
        user_wait_messages[user_id].append(wait_message.message_id)
        
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    
    # Check quota for free users
    if not await check_user_quota(user_id, update, context):
        return

    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)
    
    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    try:
        transcribed_text = await openai_utils.transcribe_audio(buf)
    except openai.error.PermissionError:
        text = get_localized_text(C.LOC_VOICE_RECOGNITION_UNAVAILABLE, user_id)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        text = get_localized_text(C.LOC_VOICE_PROCESSING_ERROR, user_id)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, C.DB_N_TRANSCRIBED_SECONDS, voice.duration + db.get_user_attribute(user_id, C.DB_N_TRANSCRIBED_SECONDS))

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images, size=config.image_size)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = get_localized_text(C.LOC_IMAGE_GENERATION_REJECTED, user_id)
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(user_id, C.DB_N_GENERATED_IMAGES, config.return_n_generated_images + db.get_user_attribute(user_id, C.DB_N_GENERATED_IMAGES))

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())
    db.set_user_attribute(user_id, C.DB_CURRENT_MODEL, C.OPENAI_MODEL_GPT_5_MINI)

    db.start_new_dialog(user_id)
    await update.message.reply_text(get_localized_text(C.LOC_STARTING_NEW_DIALOG, user_id))

    chat_mode = db.get_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE)
    if chat_mode not in config.chat_modes.keys():
        chat_mode = "ai_trainer"
        db.set_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE, chat_mode)
    
    language = db.get_user_attribute(user_id, C.DB_CURRENT_LANGUAGE) or C.DEFAULT_LANGUAGE
    welcome_message = config.chat_modes[chat_mode]['welcome_message'][language]
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text(get_localized_text(C.LOC_NOTHING_TO_CANCEL, user_id), parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int, user_id: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = get_localized_text(C.LOC_SELECT_CHAT_MODE, user_id).format(n_modes=len(config.chat_modes))

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    text, reply_markup = get_chat_mode_menu(0, user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index, user_id)
     try:
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE, chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message'][db.get_user_attribute(user_id, 'current_language') or 'en']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, C.DB_CURRENT_MODEL)
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += get_localized_text(C.LOC_SELECT_SETTINGS, user_id)

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, C.DB_CURRENT_MODEL, model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())
    
    # Check if user is admin or tester (they get detailed view)
    is_admin_or_tester = user_id in config.admin_user_ids or user_id in config.test_user_ids
    
    if is_admin_or_tester:
        # Detailed view for admins and testers
        # count total usage statistics
        total_n_spent_dollars = 0
        total_n_used_tokens = 0

        n_used_tokens_dict = db.get_user_attribute(user_id, C.DB_N_USED_TOKENS)
        n_generated_images = db.get_user_attribute(user_id, C.DB_N_GENERATED_IMAGES)
        n_transcribed_seconds = db.get_user_attribute(user_id, C.DB_N_TRANSCRIBED_SECONDS)

        details_text = get_localized_text(C.LOC_DETAILS, user_id)
        for model_key in sorted(n_used_tokens_dict.keys()):
            n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key][C.DB_N_INPUT_TOKENS], n_used_tokens_dict[model_key][C.DB_N_OUTPUT_TOKENS]
            total_n_used_tokens += n_input_tokens + n_output_tokens

            n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
            n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
            total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

            details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

        # image generation
        image_generation_n_spent_dollars = config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
        if n_generated_images != 0:
            details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

        total_n_spent_dollars += image_generation_n_spent_dollars

        # voice recognition
        voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
        if n_transcribed_seconds != 0:
            details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

        total_n_spent_dollars += voice_recognition_n_spent_dollars

        text = get_localized_text(C.LOC_SPENT, user_id).format(amount=f"{total_n_spent_dollars:.03f}")
        text += get_localized_text(C.LOC_USED_TOKENS, user_id).format(amount=total_n_used_tokens)
        text += details_text
        
        # Add user status badge
        status_badge = get_user_status_badge(user_id)
        if status_badge:
            text += f"\n\n{status_badge}"
    else:
        # Simplified view for regular users
        is_premium = is_premium_user(user_id)
        n_used_tokens_dict = db.get_user_attribute(user_id, C.DB_N_USED_TOKENS) or {}
        
        # Calculate total used tokens
        total_used_tokens = 0
        for model_key, tokens_dict in n_used_tokens_dict.items():
            total_used_tokens += tokens_dict.get(C.DB_N_INPUT_TOKENS, 0) + tokens_dict.get(C.DB_N_OUTPUT_TOKENS, 0)
        
        if is_premium:
            # Premium user - show remaining tokens and reset date
            is_test = is_test_user(user_id)
            token_limit = C.TEST_PREMIUM_MONTHLY_TOKEN_LIMIT if is_test else C.PREMIUM_MONTHLY_TOKEN_LIMIT
            remaining_tokens = max(0, token_limit - total_used_tokens)
            reset_date = db.get_user_attribute(user_id, C.DB_MONTHLY_TOKEN_RESET_DATE)
            
            text = f"📊 <b>Tokens remaining:</b> {remaining_tokens:,} / {token_limit:,}\n"
            if reset_date:
                text += f"🔄 <b>Resets on:</b> {get_formatted_date(reset_date, user_id)}"
        else:
            # Free user - show remaining tokens
            remaining_tokens = max(0, C.FREE_TOKEN_LIMIT - total_used_tokens)
            text = f"📊 <b>Tokens remaining:</b> {remaining_tokens:,} / {C.FREE_TOKEN_LIMIT:,}\n"
            text += f"\n💡 Upgrade to premium for more tokens!"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = get_localized_text(C.LOC_EDITING_NOT_SUPPORTED, update.edited_message.from_user.id)
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def show_language_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    text = get_localized_text(C.LOC_SELECT_LANGUAGE, user_id)
    
    # Language display with flags and native names
    language_display = {
        "en": "🇺🇸 English",
        "ru": "🇷🇺 Русский"
    }
    
    keyboard = []
    for language_code in config.locales.keys():
        display_name = language_display.get(language_code, language_code)
        keyboard.append([InlineKeyboardButton(display_name, callback_data=f"set_language|{language_code}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_language_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, language_code = query.data.split("|")
    db.set_user_attribute(user_id, C.DB_CURRENT_LANGUAGE, language_code)

    text = get_localized_text(C.LOC_LANGUAGE_SET, user_id).format(language=language_code)
    await context.bot.send_message(
        update.callback_query.message.chat.id,
        text,
        parse_mode=ParseMode.HTML
    )
    

def is_premium_user(user_id: int) -> bool:
    # Check if user is admin (admins have unlimited access)
    if user_id in config.admin_user_ids:
        return True
    
    # Check regular subscription status
    status = db.get_user_attribute(user_id, C.DB_SUBSCRIPTION_STATUS)
    expiry = db.get_user_attribute(user_id, C.DB_SUBSCRIPTION_EXPIRY)
    
    if status == C.SUBSCRIPTION_STATUS_PREMIUM and expiry and expiry > datetime.now():
        return True
    return False


def is_test_user(user_id: int) -> bool:
    """Check if user is a test user with special subscription parameters."""
    return user_id in config.test_user_ids


def get_user_status_badge(user_id: int) -> str:
    """Get user status badge (Admin/Tester) or empty string for regular users."""
    if user_id in config.admin_user_ids:
        return "👑 Administrator"
    elif user_id in config.test_user_ids:
        return "🧪 Tester"
    return ""


async def check_user_quota(user_id: int, update: Update, context: CallbackContext) -> bool:
    is_premium = is_premium_user(user_id)
    
    # Get current token usage
    n_used_tokens_dict = db.get_user_attribute(user_id, C.DB_N_USED_TOKENS) or {}
    
    if is_premium:
        # Determine if user is a test user
        is_test = is_test_user(user_id)
        
        # Set appropriate limits and duration
        token_limit = C.TEST_PREMIUM_MONTHLY_TOKEN_LIMIT if is_test else C.PREMIUM_MONTHLY_TOKEN_LIMIT
        reset_days = C.TEST_SUBSCRIPTION_DURATION_DAYS if is_test else 30
        
        # Check if monthly quota needs to be reset
        reset_date = db.get_user_attribute(user_id, C.DB_MONTHLY_TOKEN_RESET_DATE)
        
        # If reset_date is None or has passed, reset the quota
        if reset_date is None or datetime.now() >= reset_date:
            # Calculate next reset date
            next_reset = datetime.now() + timedelta(days=reset_days)
            db.set_user_attribute(user_id, C.DB_MONTHLY_TOKEN_RESET_DATE, next_reset)
            # Reset token usage
            db.set_user_attribute(user_id, C.DB_N_USED_TOKENS, {})
            return True
        
        # Calculate tokens used in current period
        total_tokens = 0
        for model_key, tokens_dict in n_used_tokens_dict.items():
            total_tokens += tokens_dict.get(C.DB_N_INPUT_TOKENS, 0) + tokens_dict.get(C.DB_N_OUTPUT_TOKENS, 0)
        
        # Check if premium user exceeded monthly limit
        if total_tokens >= token_limit:
            text = get_localized_text(C.LOC_PREMIUM_LIMIT_REACHED, user_id).format(
                reset_date=get_formatted_date(reset_date, user_id)
            )
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return False
        
        return True
    else:
        # Free user - check lifetime limit
        total_tokens = 0
        for model_key, tokens_dict in n_used_tokens_dict.items():
            total_tokens += tokens_dict.get(C.DB_N_INPUT_TOKENS, 0) + tokens_dict.get(C.DB_N_OUTPUT_TOKENS, 0)
            
        if total_tokens >= C.FREE_TOKEN_LIMIT:
            text = get_localized_text(C.LOC_SUBSCRIPTION_LIMIT_REACHED, user_id)
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return False
            
        return True


async def show_subscription_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    
    is_premium = is_premium_user(user_id)
    expiry = db.get_user_attribute(user_id, C.DB_SUBSCRIPTION_EXPIRY)
    
    if is_premium and expiry:
        text = get_localized_text(C.LOC_SUBSCRIPTION_STATUS_PREMIUM, user_id).format(
            expiry_date=get_formatted_date(expiry, user_id)
        )
        
        # Add user status badge if applicable
        status_badge = get_user_status_badge(user_id)
        if status_badge:
            text += f"\n\n{status_badge}"
        
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    else:
        title = get_localized_text(C.LOC_SUBSCRIPTION_TITLE, user_id)
        description = get_localized_text(C.LOC_SUBSCRIPTION_DESCRIPTION, user_id)
        invoice_description = get_localized_text(C.LOC_SUBSCRIPTION_INVOICE_DESCRIPTION, user_id)
        
        # Send a formatted message with description first
        await update.message.reply_text(description, parse_mode=ParseMode.HTML)
        
        # Use test pricing for test users
        is_test = is_test_user(user_id)
        price = C.TEST_SUBSCRIPTION_PRICE_STARS if is_test else C.SUBSCRIPTION_PRICE_STARS
        
        await context.bot.send_invoice(
            chat_id=update.message.chat_id,
            title=title,
            description=invoice_description,
            payload=C.SUBSCRIPTION_PAYLOAD_MONTHLY,
            provider_token="",  # Empty for Telegram Stars
            currency="XTR",
            prices=[LabeledPrice("Premium 1 Month", price)],
            start_parameter="premium-subscription"
        )


async def precheckout_callback(update: Update, context: CallbackContext):
    query = update.pre_checkout_query
    # Check the payload, is this from your bot?
    if query.invoice_payload != C.SUBSCRIPTION_PAYLOAD_MONTHLY:
        # Answer False if something went wrong
        user_id = update.effective_user.id
        error_text = get_localized_text(C.LOC_PAYMENT_ERROR, user_id)
        await query.answer(ok=False, error_message=error_text)
    else:
        await query.answer(ok=True)


async def successful_payment_callback(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    
    # Determine if user is a test user
    is_test = is_test_user(user_id)
    subscription_duration = C.TEST_SUBSCRIPTION_DURATION_DAYS if is_test else C.SUBSCRIPTION_DURATION_DAYS
    
    # Calculate new expiry date
    current_expiry = db.get_user_attribute(user_id, C.DB_SUBSCRIPTION_EXPIRY)
    if current_expiry and current_expiry > datetime.now():
        new_expiry = current_expiry + timedelta(days=subscription_duration)
    else:
        new_expiry = datetime.now() + timedelta(days=subscription_duration)
        
    # Update user subscription
    db.set_user_attribute(user_id, C.DB_SUBSCRIPTION_STATUS, C.SUBSCRIPTION_STATUS_PREMIUM)
    db.set_user_attribute(user_id, C.DB_SUBSCRIPTION_EXPIRY, new_expiry)
    
    # Set monthly token reset date
    monthly_reset_date = datetime.now() + timedelta(days=subscription_duration)
    db.set_user_attribute(user_id, C.DB_MONTHLY_TOKEN_RESET_DATE, monthly_reset_date)
    
    # Reset token usage for new subscription period
    db.set_user_attribute(user_id, C.DB_N_USED_TOKENS, {})
    
    # Add to history
    payment_info = {
        "date": datetime.now(),
        "amount": update.message.successful_payment.total_amount,
        "currency": update.message.successful_payment.currency,
        "telegram_payment_charge_id": update.message.successful_payment.telegram_payment_charge_id,
        "provider_payment_charge_id": update.message.successful_payment.provider_payment_charge_id
    }
    
    history = db.get_user_attribute(user_id, C.DB_SUBSCRIPTION_HISTORY) or []
    history.append(payment_info)
    db.set_user_attribute(user_id, C.DB_SUBSCRIPTION_HISTORY, history)
    
    text = get_localized_text(C.LOC_SUBSCRIPTION_SUCCESS, user_id).format(
        expiry_date=get_formatted_date(new_expiry, user_id)
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    
    # Refresh menu if needed or just confirm
    # For now just confirmation message is enough


# Profile Management Handlers

PROFILE_FIELDS_ORDER = [
    C.PROFILE_HEIGHT,
    C.PROFILE_WEIGHT,
    C.PROFILE_FITNESS_LEVEL,
    C.PROFILE_GOALS,
    C.PROFILE_GENDER
]

async def prompt_next_empty_profile_field(user_id: int, context: CallbackContext, update: Update):
    profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
    
    for field in PROFILE_FIELDS_ORDER:
        if not profile.get(field):
            # Found empty field, prompt for it
            context.user_data[C.CONTEXT_PROFILE_FIELD_EDITING] = field
            
            if field == C.PROFILE_FITNESS_LEVEL:
                text = get_localized_text(C.LOC_PROFILE_SELECT_FITNESS_LEVEL, user_id)
                keyboard = [
                    [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_BEGINNER, user_id), callback_data="profile_set|fitness_level|beginner")],
                    [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_INTERMEDIATE, user_id), callback_data="profile_set|fitness_level|intermediate")],
                    [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_ADVANCED, user_id), callback_data="profile_set|fitness_level|advanced")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                if update.callback_query:
                    await update.callback_query.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
                else:
                    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
                    
            elif field == C.PROFILE_GENDER:
                text = get_localized_text(C.LOC_PROFILE_SELECT_GENDER, user_id)
                keyboard = [
                    [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_MALE, user_id), callback_data="profile_set|gender|male")],
                    [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_FEMALE, user_id), callback_data="profile_set|gender|female")],
                    [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_OTHER, user_id), callback_data="profile_set|gender|other")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                if update.callback_query:
                    await update.callback_query.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
                else:
                    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
            else:
                if field == C.PROFILE_HEIGHT:
                    text = get_localized_text(C.LOC_PROFILE_ENTER_HEIGHT, user_id)
                elif field == C.PROFILE_WEIGHT:
                    text = get_localized_text(C.LOC_PROFILE_ENTER_WEIGHT, user_id)
                elif field == C.PROFILE_GOALS:
                    text = get_localized_text(C.LOC_PROFILE_ENTER_GOALS, user_id)
                
                if update.callback_query:
                    await update.callback_query.message.reply_text(text, parse_mode=ParseMode.HTML)
                else:
                    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return

    # All fields filled, show profile summary
    await show_profile_handle(update, context)


async def show_profile_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, C.DB_LAST_INTERACTION, datetime.now())

    profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
    
    # Format profile data
    height = f"{profile[C.PROFILE_HEIGHT]} cm" if profile.get(C.PROFILE_HEIGHT) else "-"
    text = get_localized_text(C.LOC_PROFILE_TITLE, user_id) + "\n\n"
    text += get_localized_text(C.LOC_PROFILE_CURRENT, user_id).format(
        height=f"{profile.get(C.PROFILE_HEIGHT, '-')} cm" if profile.get(C.PROFILE_HEIGHT) else "-",
        weight=f"{profile.get(C.PROFILE_WEIGHT, '-')} kg" if profile.get(C.PROFILE_WEIGHT) else "-",
        fitness_level=profile.get(C.PROFILE_FITNESS_LEVEL, "-"),
        goals=profile.get(C.PROFILE_GOALS, "-"),
        gender=profile.get(C.PROFILE_GENDER, "-")
    )
    
    text += "\n\n" + get_localized_text(C.LOC_PROFILE_SELECT_FIELD, user_id)
    
    keyboard = [
        [InlineKeyboardButton(get_localized_text(C.LOC_BUTTON_HEIGHT, user_id), callback_data="profile_edit|height")],
        [InlineKeyboardButton(get_localized_text(C.LOC_BUTTON_WEIGHT, user_id), callback_data="profile_edit|weight")],
        [InlineKeyboardButton(get_localized_text(C.LOC_BUTTON_FITNESS_LEVEL, user_id), callback_data="profile_edit|fitness_level")],
        [InlineKeyboardButton(get_localized_text(C.LOC_BUTTON_GOALS, user_id), callback_data="profile_edit|goals")],
        [InlineKeyboardButton(get_localized_text(C.LOC_BUTTON_GENDER, user_id), callback_data="profile_edit|gender")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def profile_edit_callback_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id
    query = update.callback_query
    await query.answer()

    _, field = query.data.split("|")
    context.user_data[C.CONTEXT_PROFILE_FIELD_EDITING] = field
    
    if field == C.PROFILE_FITNESS_LEVEL:
        text = get_localized_text(C.LOC_PROFILE_SELECT_FITNESS_LEVEL, user_id)
        keyboard = [
            [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_BEGINNER, user_id), callback_data="profile_set|fitness_level|beginner")],
            [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_INTERMEDIATE, user_id), callback_data="profile_set|fitness_level|intermediate")],
            [InlineKeyboardButton(get_localized_text(C.LOC_FITNESS_LEVEL_ADVANCED, user_id), callback_data="profile_set|fitness_level|advanced")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif field == C.PROFILE_GENDER:
        text = get_localized_text(C.LOC_PROFILE_SELECT_GENDER, user_id)
        keyboard = [
            [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_MALE, user_id), callback_data="profile_set|gender|male")],
            [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_FEMALE, user_id), callback_data="profile_set|gender|female")],
            [InlineKeyboardButton(get_localized_text(C.LOC_GENDER_OTHER, user_id), callback_data="profile_set|gender|other")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        if field == C.PROFILE_HEIGHT:
            text = get_localized_text(C.LOC_PROFILE_ENTER_HEIGHT, user_id)
        elif field == C.PROFILE_WEIGHT:
            text = get_localized_text(C.LOC_PROFILE_ENTER_WEIGHT, user_id)
        elif field == C.PROFILE_GOALS:
            text = get_localized_text(C.LOC_PROFILE_ENTER_GOALS, user_id)
        
        await context.bot.send_message(user_id, text, parse_mode=ParseMode.HTML)
        try:
            await query.delete_message()
        except:
            pass


async def profile_set_callback_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id
    query = update.callback_query
    await query.answer()

    _, field, value = query.data.split("|")
    
    profile = db.get_user_attribute(user_id, C.DB_USER_PROFILE) or {}
    profile[field] = value
    db.set_user_attribute(user_id, C.DB_USER_PROFILE, profile)
    
    text = get_localized_text(C.LOC_PROFILE_UPDATED, user_id)
    await query.edit_message_text(text, parse_mode=ParseMode.HTML)
    await prompt_next_empty_profile_field(user_id, context, update)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        if update and update.effective_chat:
            for message_chunk in split_text_into_chunks(message, 4096):
                try:
                    await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
                except telegram.error.BadRequest:
                    # answer has invalid characters, so we send it without parse_mode
                    await context.bot.send_message(update.effective_chat.id, message_chunk)
    except Exception as e:
        logger.error(f"Error in error handler: {e}")

async def post_init(application: Application):
    # Set commands for English (default)
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/retry", "Re-generate response"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/balance", "Show balance"),
        BotCommand("/language", "Select language"),
        BotCommand("/profile", "Manage your profile"),
        BotCommand("/subscribe", "Premium subscription"),
        BotCommand("/help", "Show help message"),
    ])
    
    # Set commands for Russian
    await application.bot.set_my_commands([
        BotCommand("/new", "Начать новый диалог"),
        BotCommand("/retry", "Перегенерировать ответ"),
        BotCommand("/settings", "Показать настройки"),
        BotCommand("/balance", "Показать баланс"),
        BotCommand("/language", "Выбрать язык"),
        BotCommand("/profile", "Управление профилем"),
        BotCommand("/subscribe", "Премиум подписка"),
        BotCommand("/help", "Показать помощь"),
    ], language_code="ru")

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .connect_timeout(60)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(60)
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.VIDEO & ~filters.COMMAND & user_filter, unsupport_message_handle))
    application.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND & user_filter, unsupport_message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))
    application.add_handler(CommandHandler("language", show_language_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_language_handle, pattern="^set_language"))
    
    application.add_handler(CommandHandler("profile", show_profile_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(profile_edit_callback_handle, pattern="^profile_edit"))
    application.add_handler(CallbackQueryHandler(profile_set_callback_handle, pattern="^profile_set"))

    application.add_handler(CommandHandler("subscribe", show_subscription_handle, filters=user_filter))
    application.add_handler(CommandHandler("subscription", show_subscription_handle, filters=user_filter))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
