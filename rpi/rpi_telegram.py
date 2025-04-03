import telegram
import asyncio
import datetime

# Telegram bot configuration
BOT_TOKEN = '7782063988:AAGAqTPsiVU_Vqhkkw1KvqjhrdKwONeaB0o'
bot = telegram.Bot(token=BOT_TOKEN)

# Chat IDs
anomaly_detection_channel_id = -1002589162188
slms_priority_1_channel_id = -1002531314734
slms_priority_2_channel_id = -1002500092362
slms_priority_3_channel_id = -1002647870078

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(chat_id=chat_id, text=text)

async def send_document(document, chat_id):
    async with bot:
        await bot.send_document(document=document, chat_id=chat_id)

async def send_photo(photo, chat_id, caption=None):
    async with bot:
        await bot.send_photo(photo=photo, chat_id=chat_id, caption=caption)

async def send_video(video, chat_id, caption=None):
    async with bot:
        await bot.send_video(video=video, chat_id=chat_id, caption=caption)

async def send_start_msg():
    startup_msg = f"🚀 Anomaly Detection System Started\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💻 Device: RPi 5"
    await send_message(startup_msg, anomaly_detection_channel_id)

async def send_shutdown_msg(stats_msg):
    shutdown_msg = f"🛑 Anomaly Detection System Stopped\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💻 Device: RPi 5\n"
    await send_message(shutdown_msg + stats_msg, anomaly_detection_channel_id)

async def send_anomaly_msg(anomaly_msg):
    # anomaly_msg = f"⚠️ Anomaly Detected\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💻 Device: RPi 5\n{anomaly_msg}"
    await send_message(anomaly_msg, anomaly_detection_channel_id)