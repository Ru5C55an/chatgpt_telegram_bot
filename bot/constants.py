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
