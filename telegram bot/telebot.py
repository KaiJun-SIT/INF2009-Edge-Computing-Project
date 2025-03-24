import telegram
import asyncio

bot = telegram.Bot(token='token_id')

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)


async def send_document(document, chat_id):
    async with bot:
        await bot.send_document(document=document, chat_id=chat_id)


async def send_photo(photo, chat_id):
    async with bot:
        await bot.send_photo(photo=photo, chat_id=chat_id)


async def send_video(video, chat_id):
    async with bot:
        await bot.send_video(video=video, chat_id=chat_id)
        
anomaly_detection_channel_id = -1002589162188
slms_priority_1_channel_id = -1002531314734
slms_priority_2_channel_id = -1002500092362
slms_priority_3_channel_id = -1002647870078

# Text Messages
# bot.send_message(chat_id=anomaly_detection_channel_id, text='Hello, World!')


# Different File Types
# bot.send_photo(chat_id=channel_id, photo=open('/path/to/photo.jpg', 'rb'))
# bot.send_document(chat_id=channel_id, document=open('/path/to/document.pdf', 'rb'))
# bot.send_video(chat_id=channel_id, video=open('/path/to/video.mp4', 'rb'))

async def main():
    # Sending a message
    await send_message(text="anomaly detected", chat_id=anomaly_detection_channel_id)
    await send_message(text="priority 1 detected", chat_id=slms_priority_1_channel_id)
    await send_message(text="priority 2 detected", chat_id=slms_priority_2_channel_id)
    await send_message(text="priority 3 detected", chat_id=slms_priority_3_channel_id)



    # Sending a document
    # await send_document(document=open('/path/to/document.pdf', 'rb'), chat_id=chat_id)

    # Sending a photo
    # await send_photo(photo=open('/path/to/photo.jpg', 'rb'), chat_id=chat_id)

    # Sending a video
    # await send_video(video=open('path/to/video.mp4', 'rb'), chat_id=chat_id)


if __name__ == '__main__':
    asyncio.run(main())
