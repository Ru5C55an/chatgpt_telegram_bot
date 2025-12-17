"""
Constants for database attribute keys and context data keys.
This module centralizes all string keys used throughout the bot to avoid typos and improve maintainability.
"""

# Database user attribute keys
DB_CURRENT_DIALOG_ID = "current_dialog_id"
DB_CURRENT_CHAT_MODE = "current_chat_mode"
DB_CURRENT_MODEL = "current_model"
DB_CURRENT_LANGUAGE = "current_language"
DB_USER_PROFILE = "user_profile"
DB_LAST_INTERACTION = "last_interaction"
DB_FIRST_SEEN = "first_seen"
DB_N_USED_TOKENS = "n_used_tokens"
DB_N_INPUT_TOKENS = "n_input_tokens"
DB_N_OUTPUT_TOKENS = "n_output_tokens"
DB_N_GENERATED_IMAGES = "n_generated_images"
DB_N_TRANSCRIBED_SECONDS = "n_transcribed_seconds"

# Dialog collection keys
DB_DIALOG_ID = "_id"
DB_DIALOG_USER_ID = "user_id"
DB_DIALOG_CHAT_MODE = "chat_mode"
DB_DIALOG_START_TIME = "start_time"
DB_DIALOG_MODEL = "model"
DB_DIALOG_MESSAGES = "messages"

# User profile field keys
PROFILE_HEIGHT = "height"
PROFILE_WEIGHT = "weight"
PROFILE_FITNESS_LEVEL = "fitness_level"
PROFILE_GOALS = "goals"
PROFILE_GENDER = "gender"

# Context user_data keys
CONTEXT_PROFILE_FIELD_EDITING = "profile_field_editing"

# Default values
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_LANGUAGE = "en"
DEFAULT_CHAT_MODE = "ai_trainer"

# Subscription keys
DB_SUBSCRIPTION_STATUS = "subscription_status"
DB_SUBSCRIPTION_EXPIRY = "subscription_expiry"
DB_SUBSCRIPTION_HISTORY = "subscription_history"
DB_MONTHLY_TOKEN_RESET_DATE = "monthly_token_reset_date"

# Subscription settings
SUBSCRIPTION_STATUS_FREE = "free"
SUBSCRIPTION_STATUS_PREMIUM = "premium"
SUBSCRIPTION_PRICE_STARS = 1000
SUBSCRIPTION_DURATION_DAYS = 30
SUBSCRIPTION_PAYLOAD_MONTHLY = "premium_subscription_monthly"

# Limits
FREE_TOKEN_LIMIT = 5000
PREMIUM_MONTHLY_TOKEN_LIMIT = 4_100_000

# Test user subscription settings
TEST_SUBSCRIPTION_PRICE_STARS = 1
TEST_SUBSCRIPTION_DURATION_DAYS = 1
TEST_PREMIUM_MONTHLY_TOKEN_LIMIT = 15_000

# OpenAI Models
OPENAI_MODEL_GPT_5_MINI = "gpt-5-mini-2025-08-07"
OPENAI_MODEL_GPT_4_VISION = "gpt-4-vision-preview"
OPENAI_MODEL_GPT_4O = "gpt-4o"
OPENAI_MODEL_DAVINCI = "text-davinci-003"
OPENAI_MODEL_WHISPER = "whisper-1"

# OpenAI Roles
OPENAI_ROLE_SYSTEM = "system"
OPENAI_ROLE_USER = "user"
OPENAI_ROLE_ASSISTANT = "assistant"

# Localization keys
LOC_WELCOME = "welcome"
LOC_HELP = "help"
LOC_HELP_GROUP_CHAT = "help_group_chat"
LOC_NO_MESSAGE_TO_RETRY = "no_message_to_retry"
LOC_IMAGES_PROCESSING_NOT_AVAILABLE = "images_processing_not_available"
LOC_STARTING_NEW_DIALOG_TIMEOUT = "starting_new_dialog_timeout"
LOC_ERROR_MESSAGE = "error_message"
LOC_UNSUPPORTED_MESSAGE_TYPE = "unsupported_message_type"
LOC_PROFILE_UPDATED = "profile_updated"
LOC_PROFILE_INVALID_NUMBER = "profile_invalid_number"
LOC_EMPTY_MESSAGE = "empty_message"
LOC_CONTEXT_REMOVED_FIRST = "context_removed_first"
LOC_CONTEXT_REMOVED_MULTIPLE = "context_removed_multiple"
LOC_CANCELED = "canceled"
LOC_WAIT_FOR_REPLY = "wait_for_reply"
LOC_VOICE_RECOGNITION_UNAVAILABLE = "voice_recognition_unavailable"
LOC_VOICE_PROCESSING_ERROR = "voice_processing_error"
LOC_IMAGE_GENERATION_REJECTED = "image_generation_rejected"
LOC_STARTING_NEW_DIALOG = "starting_new_dialog"
LOC_NOTHING_TO_CANCEL = "nothing_to_cancel"
LOC_SELECT_CHAT_MODE = "select_chat_mode"
LOC_SELECT_SETTINGS = "select_settings"
LOC_DETAILS = "details"
LOC_SPENT = "spent"
LOC_USED_TOKENS = "used_tokens"
LOC_EDITING_NOT_SUPPORTED = "editing_not_supported"
LOC_SELECT_LANGUAGE = "select_language"
LOC_LANGUAGE_SET = "language_set"
LOC_PREMIUM_LIMIT_REACHED = "premium_limit_reached"
LOC_SUBSCRIPTION_LIMIT_REACHED = "subscription_limit_reached"
LOC_SUBSCRIPTION_STATUS_PREMIUM = "subscription_status_premium"
LOC_SUBSCRIPTION_TITLE = "subscription_title"
LOC_SUBSCRIPTION_DESCRIPTION = "subscription_description"
LOC_PAYMENT_ERROR = "payment_error"
LOC_SUBSCRIPTION_SUCCESS = "subscription_success"
LOC_PROFILE_SELECT_FITNESS_LEVEL = "profile_select_fitness_level"
LOC_FITNESS_LEVEL_BEGINNER = "fitness_level_beginner"
LOC_FITNESS_LEVEL_INTERMEDIATE = "fitness_level_intermediate"
LOC_FITNESS_LEVEL_ADVANCED = "fitness_level_advanced"
LOC_PROFILE_SELECT_GENDER = "profile_select_gender"
LOC_GENDER_MALE = "gender_male"
LOC_GENDER_FEMALE = "gender_female"
LOC_GENDER_OTHER = "gender_other"
LOC_PROFILE_ENTER_HEIGHT = "profile_enter_height"
LOC_PROFILE_ENTER_WEIGHT = "profile_enter_weight"
LOC_PROFILE_ENTER_GOALS = "profile_enter_goals"
LOC_PROFILE_TITLE = "profile_title"
LOC_PROFILE_EMPTY = "profile_empty"
LOC_PROFILE_CURRENT = "profile_current"
LOC_PROFILE_SELECT_FIELD = "profile_select_field"
LOC_BUTTON_HEIGHT = "button_height"
LOC_BUTTON_WEIGHT = "button_weight"
LOC_BUTTON_FITNESS_LEVEL = "button_fitness_level"
LOC_BUTTON_GOALS = "button_goals"
LOC_BUTTON_GENDER = "button_gender"
LOC_OPENAI_RATE_LIMIT_ERROR = "openai_rate_limit_error"
