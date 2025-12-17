from typing import Optional, Any

import pymongo
import uuid
from datetime import datetime

import config
import constants as C


class Database:
    def __init__(self):
        self.client = pymongo.MongoClient(config.mongodb_uri)
        self.db = self.client["chatgpt_telegram_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        if self.user_collection.count_documents({"_id": user_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"User {user_id} does not exist")
            else:
                return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,

            "username": username,
            "first_name": first_name,
            "last_name": last_name,

            C.DB_LAST_INTERACTION: datetime.now(),
            C.DB_FIRST_SEEN: datetime.now(),

            C.DB_CURRENT_DIALOG_ID: None,
            C.DB_CURRENT_CHAT_MODE: C.DEFAULT_CHAT_MODE,
            C.DB_CURRENT_MODEL: C.DEFAULT_MODEL,
            C.DB_CURRENT_LANGUAGE: C.DEFAULT_LANGUAGE,
            C.DB_USER_PROFILE: {
                C.PROFILE_HEIGHT: None,
                C.PROFILE_WEIGHT: None,
                C.PROFILE_FITNESS_LEVEL: None,
                C.PROFILE_GOALS: None,
                C.PROFILE_GENDER: None,
            },

            C.DB_N_USED_TOKENS: {},

            C.DB_N_GENERATED_IMAGES: 0,
            C.DB_N_TRANSCRIBED_SECONDS: 0.0,  # voice message transcription

            # Subscription fields
            C.DB_SUBSCRIPTION_STATUS: C.SUBSCRIPTION_STATUS_FREE,
            C.DB_SUBSCRIPTION_EXPIRY: None,
            C.DB_SUBSCRIPTION_HISTORY: [],
            C.DB_MONTHLY_TOKEN_RESET_DATE: None
        }

        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)

    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = str(uuid.uuid4())
        dialog_dict = {
            C.DB_DIALOG_ID: dialog_id,
            C.DB_DIALOG_USER_ID: user_id,
            C.DB_DIALOG_CHAT_MODE: self.get_user_attribute(user_id, C.DB_CURRENT_CHAT_MODE),
            C.DB_DIALOG_START_TIME: datetime.now(),
            C.DB_DIALOG_MODEL: self.get_user_attribute(user_id, C.DB_CURRENT_MODEL),
            C.DB_DIALOG_MESSAGES: []
        }

        # add new dialog
        self.dialog_collection.insert_one(dialog_dict)

        # update user's current dialog
        self.user_collection.update_one(
            {"_id": user_id},
            {"$set": {C.DB_CURRENT_DIALOG_ID: dialog_id}}
        )

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})

        if key not in user_dict:
            return None

        return user_dict[key]

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.user_collection.update_one({"_id": user_id}, {"$set": {key: value}})

    def update_n_used_tokens(self, user_id: int, model: str, n_input_tokens: int, n_output_tokens: int):
        n_used_tokens_dict = self.get_user_attribute(user_id, C.DB_N_USED_TOKENS)

        if model in n_used_tokens_dict:
            n_used_tokens_dict[model][C.DB_N_INPUT_TOKENS] += n_input_tokens
            n_used_tokens_dict[model][C.DB_N_OUTPUT_TOKENS] += n_output_tokens
        else:
            n_used_tokens_dict[model] = {
                C.DB_N_INPUT_TOKENS: n_input_tokens,
                C.DB_N_OUTPUT_TOKENS: n_output_tokens
            }

        self.set_user_attribute(user_id, C.DB_N_USED_TOKENS, n_used_tokens_dict)

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, C.DB_CURRENT_DIALOG_ID)

        dialog_dict = self.dialog_collection.find_one({C.DB_DIALOG_ID: dialog_id, C.DB_DIALOG_USER_ID: user_id})
        return dialog_dict[C.DB_DIALOG_MESSAGES]

    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, C.DB_CURRENT_DIALOG_ID)

        self.dialog_collection.update_one(
            {C.DB_DIALOG_ID: dialog_id, C.DB_DIALOG_USER_ID: user_id},
            {"$set": {C.DB_DIALOG_MESSAGES: dialog_messages}}
        )
