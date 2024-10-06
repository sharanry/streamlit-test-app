from datetime import time

from backend.recsys import Arm, ChatItem, Recommender, SamplerType

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
    ChatItem({"sender": "Assistant",
             "message": "Here is the weather for today: Temperature: 10°C, Humidity: 50%, Wind: 10 km/h, Rain: 10%."}),
    ChatItem({"sender": "Assistant",
             "message": "Here is the news for today: Ukraine News - Ukraine is in a state of war with Russia."})
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

    arms.append(Arm('xkcd arm', {}, sampler_type=SamplerType.XKCD, init_score=5.0))
    arms.append(Arm('arxiv arm', {'topic': 'quantum computing'}, sampler_type=SamplerType.ARXIV, init_score=5.0))
    rec = Recommender(arms)
    return rec
