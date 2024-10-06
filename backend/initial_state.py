from datetime import time

from backend.recsys import Arm, ChatItem, GNewsItem, XKCDItem, Recommender, SamplerType

# Convert the thread into ChatItem objects
thread = [
    ChatItem({"sender": "Assistant", "message": "Hello! I'm your personal assistant. I can help you with keeping track of news, share memes, and even send weather updates! Let me know how I can assist you!"}),
    ChatItem(
        {"sender": "User", "message": "Give me a brief on the weather every morning."}),
    ChatItem({"sender": "Assistant",
             "message": "I will send you a message every morning with the weather."}),
    ChatItem(
        {"sender": "User", "message": "I am also interested in news on Ukraine."}),
    ChatItem({"sender": "Assistant",
             "message": "I will send you a message whenever there is important news on Ukraine."}),
    GNewsItem({
             "title": "Behind Trump’s Views on Ukraine: Putin’s Gambit and a Political Grudge",
             "url": "https://www.nytimes.com/2024/10/05/us/politics/trump-putin-ukraine.html",
             "description": "Behind Trump’s Views on Ukraine: Putin’s Gambit and a Political Grudge"
             }),
    XKCDItem({
        "number": 959,
        "title": "Caroling",
        "altText": "This is a funny comic.",
        "imageLink": "https://imgs.xkcd.com/comics/caroling.png",
        # "image_name": "funny_comic",
        "link": "https://xkcd.com/1234/"
    })
]


memory = [
    ""
]

topics = [{
    "topic": "Weather",
    "weight": 10,
    "time": time(hour=8)
}, {
    "topic": "Ukraine",
    "weight": 7,
    "time": time(hour=10)
}, {
    "topic": "Cricket",
    "weight": 5,
    "time": time(hour=8)
}]


def new_recommender():
    # initialize arms and bandit
    arms = []
    for i, topic in enumerate(topics):
        arms.append(Arm(f"Arm_{i}({topic})", {'query': topic},
                    sampler_type=SamplerType.GNEWS, init_score=4.0))

    arms.append(Arm('xkcd arm', {}, sampler_type=SamplerType.XKCD, init_score=7.0))
    rec = Recommender(arms)
    return rec
