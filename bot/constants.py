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
DB_N_USED_TOKENS = "n_used_tokens"
DB_N_GENERATED_IMAGES = "n_generated_images"
DB_N_TRANSCRIBED_SECONDS = "n_transcribed_seconds"

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

# Subscription settings
SUBSCRIPTION_STATUS_FREE = "free"
SUBSCRIPTION_STATUS_PREMIUM = "premium"
SUBSCRIPTION_PRICE_STARS = 1
SUBSCRIPTION_DURATION_DAYS = 1
SUBSCRIPTION_PAYLOAD_MONTHLY = "premium_subscription_monthly"
