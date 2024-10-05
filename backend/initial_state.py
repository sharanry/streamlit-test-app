from datetime import time

from backend.recsys import Arm, Bandit, Recommender, SamplerType

thread = [
    {
        "role": "Assistant",
        "message": "Hello! I'm your personal assistant. I can help you with keeping track of news, share memes, and even send weather updates! Let me know how I can assist you!"
    },
    {
        "role": "User",
        "message": "Give me a breif on the weather every morning."
    },
    {
        "role": "Assistant",
        "message": "I will send you a message every morning with the weather. "
    },
    {
        "role": "User",
        "message": "I am also interested in news on Ukraine."
    },
    {
        "role": "Assistant",
        "message": "I will send you a message whenever there is important news on Ukraine."
    },
    {
        "role": "Assistant",
        "message": "Here is the weather for today: ",
        "weather": {
            "temperature": 10,
            "humidity": 50,
            "wind": "10 km/h",
            "rain": "10%"
        }
    },
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
    rec = Recommender(arms)
    return rec
