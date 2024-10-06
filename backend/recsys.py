import asyncio
from tfl.client import Client as TFLClient
from tfl.api_token import ApiToken
import xkcd
from typing import Dict, Union, List
from gnews import GNews
import random
import numpy as np
from typing import List
from enum import Enum
import arxiv


class SamplerType(Enum):
    GNEWS = 1
    XKCD = 2
    ARXIV = 3


class Item:
    def __init__(self, sample_result: dict):
        self.sample_result = sample_result

    def __str__(self):
        return str(self.sample_result)

    def __repr__(self):
        return f"Item(sample_result={self.sample_result})"


class ChatItem(Item):
    def __init__(self, sample_result: dict):
        self.sender = sample_result.get("sender", "Unknown sender")
        self.message = sample_result.get("message", "no message")

    def __str__(self):
        return f"{self.sender} send a message: {self.message}"


class GNewsItem(Item):
    def __init__(self, sample_result: dict = {}):
        self.title = sample_result.get("title", "An untitled article")
        self.description = sample_result.get(
            "description", "No description available")
        self.date = sample_result.get("published date", "an unknown date")
        self.publisher_name = sample_result.get(
            "publisher", {}).get("title", "an unknown publisher")
        self.url = sample_result.get("url", "No url available")

    def __str__(self):
        return (
            f"'{self.title}', published by {self.publisher_name} on {
                self.date}, covers the following: {self.description}."
        )


class XKCDItem(Item):
    def __init__(self, sample_result: dict = {}):
        self.number = sample_result.get("number", "an unknown number")
        self.title = sample_result.get("title", "An untitled comic")
        self.alt_text = sample_result.get("altText", "No alt text available")
        self.image_link = sample_result.get("imageLink", "No image available")
        self.image_name = sample_result.get(
            "imageName", "No image name available")
        self.link = sample_result.get("link", "No link available")

    def __str__(self):
        return (
            f"XKCD comic #{self.number}, titled '{
                self.title}', can be viewed at {self.link}. "
            f"The comic features an image named '{
                self.image_name}', available at {self.image_link}. "
            f"Alt text: '{self.alt_text}'."
        )


class ArxivItem:
    def __init__(self, paper):
        self.title = paper.title or "Untitled paper"
        self.authors = ", ".join(
            [author.name for author in paper.authors]) or "Unknown authors"
        self.summary = paper.summary or "No summary available"
        self.published = paper.published.strftime(
            "%B %d, %Y") if paper.published else "Unknown publication date"
        self.pdf_url = paper.pdf_url or "No PDF available"
        self.arxiv_url = paper.entry_id or "No arXiv URL available"

    def __str__(self):
        return (
            f"'{self.title}' was authored by {
                self.authors} and published on {self.published}. "
            f"Here is a brief summary: {self.summary[:200]}... "
            f"Read more: {self.arxiv_url} or download the PDF at {
                self.pdf_url}."
        )


class Sampler:
    def __init__(self):
        raise NotImplementedError()

    def sample(self, params: Dict[str, Union[str, float]] = {}):
        """
        Sample function that processes the input parameter string and interacts with the API.

        Parameters:
        params (Dict): Dict with arguments

        Returns:
        dict: A dictionary containing the response from the API.
        """
        raise NotImplementedError()


class GNewsSampler(Sampler):
    def __init__(self, max_results=50):
        # Instantiate the Google News client once
        self.google_news = GNews(max_results=max_results)

    def sample(self, params: Dict[str, Union[str, float]] = {'query': 'World News'}):
        """
        Get news articles based on the 'query' parameter from the params dictionary.
        """
        query = params.get('query', '')
        print(query)
        news = self.google_news.get_news(query['topic'])

        if news:
            return GNewsItem(random.choice(news))
        else:
            raise Exception("No GNews articles found.")


class XKCDSampler(Sampler):
    def __init__(self):
        pass  # No need for an API key

    def sample(self, params: Dict[str, Union[str, float]] = {}):
        """
        Get a random XKCD comic. The 'params' dictionary is ignored as no parameters are required.
        """
        comic = xkcd.getRandomComic()
        return XKCDItem(vars(comic))  # Return the comic data


class ArxivSampler(Sampler):
    def __init__(self):
        # Initialize the arxiv client
        self.client = arxiv.Client()

    def sample(self, params: Dict[str, Union[str, float]] = {}):
        """
        Get a random paper from Arxiv. The params has the topic of interest as 'topic'.
        """
        # Extract the topic from params, default to "artificial intelligence" if not provided
        topic = params.get('topic', 'artificial intelligence')

        # Define the search query for Arxiv
        search = arxiv.Search(
            query=topic,
            max_results=100,  # Fetch up to 100 papers to allow for random sampling
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        # Get all results from the search
        all_results = list(self.client.results(search))

        if not all_results:
            return f"No papers found for the topic: {topic}"

        # Select a random paper from the search results
        random_paper = random.choice(all_results)

        # Return an ArxivItem
        return ArxivItem(random_paper)


class Arm:
    def __init__(self, name: str, params: Dict[str, Union[str, float]], sampler_type: Sampler, init_score=5, decay_rate=0.1):
        """
        Initializes an Arm with a name, an initial score, and a decay rate.

        Args:
            name (str): Name of the arm.
            init_score (float): Initial score of the arm, fixed between 0 and 10.
            decay_rate (float): The rate at which the score decays towards 5 with each pull.
        """
        self.name = name
        self.params = params
        self.sampler_type = sampler_type
        self.score = init_score  # Fixed score between 0 and 10
        self.pulls = 0  # Track how many times the arm has been pulled
        self.decay_rate = decay_rate  # Decay rate towards 5 on pull

    def sample_arm(self):
        """Simulate pulling the arm. Decay the score towards 5."""
        self.pulls += 1
        # Exponential decay towards 5
        self.score = self.score + self.decay_rate * (5 - self.score)

    def __str__(self):
        """String representation of the Arm object."""
        return f"Arm {self.name} has score {self.score:.2f} and has been pulled {self.pulls} times"


class Bandit:
    def __init__(self, arms: List[Arm] = [], alpha: float = 2.0, base_arms: List[Arm] = []):
        """
        Initializes the Bandit algorithm with a set of arms using exponential sampling based on score.

        Args:
            arms (List[Arm]): List of Arm objects.
            alpha (float): Exponential scaling factor for sampling probabilities.
        """
        self.arms = arms  # A list of Arm objects
        self.base_arms = base_arms
        self.alpha = alpha  # Scaling factor for exponential distribution

    def get_valid_arms(self):
        return self.arms+self.base_arms

    def get_probabilities(self) -> List[float]:
        """Calculates the probabilities of the arms based on their scores following an exponential distribution."""
        valid_scores = [arm.score **
                        self.alpha for arm in self.get_valid_arms() if arm.score >= 4]

        if not valid_scores:
            raise ValueError("No arms with score >= 4 to sample from.")

        total_score = sum(valid_scores)
        probabilities = [score/total_score for score in valid_scores]
        return probabilities

    def select_arm(self) -> Arm:
        """Selects and returns the arm to pull based on the exponential distribution of the scores."""
        # Filter out arms with scores less than 4
        valid_arms = [arm for arm in self.get_valid_arms() if arm.score >= 4]

        if not valid_arms:
            raise ValueError(
                "No valid arms with score >= 4 available for selection.")

        probabilities = self.get_probabilities()
        selected_arm = random.choices(
            valid_arms, weights=probabilities, k=1)[0]
        return selected_arm

    def pull_and_decay(self, arm: Arm):
        """Pulls the arm and applies the decay towards 5."""
        arm.sample_arm()


class Recommender:
    def __init__(self, base_arms=List[Arm]):
        self.bandit = Bandit(base_arms=base_arms)
        self.samplers = {
            SamplerType.GNEWS: GNewsSampler(),
            SamplerType.XKCD: XKCDSampler(),
            SamplerType.ARXIV: ArxivSampler(),
        }

    def add_arm(self, arm: Arm):
        self.bandit.arms.append(arm)

    def remove_arm(self, arm_name: str):
        self.bandit.arms = [
            arm for arm in self.bandit.arms if arm.name != arm_name]

    def sample(self) -> Item:
        # Select an arm using the bandit algorithm
        selected_arm = self.bandit.select_arm()
        selected_arm.sample_arm()

        # Get the corresponding sampler
        sampler_type = selected_arm.sampler_type
        sampler = self.samplers[sampler_type]

        sample = sampler.sample(selected_arm.params)

        return sample

    def get_arms(self) -> dict:
        """
        Returns a dictionary representation of all current arms and their configurations.
        """
        arms_config = []
        for arm in self.bandit.get_valid_arms():
            arms_config.append({
                'name': arm.name,
                'params': arm.params,
                'sampler_type': arm.sampler_type,
                'score': arm.score,
            })
        return {'arms': arms_config}

    def update_arms(self, new_arms_config: dict) -> bool:
        """
        Update the current set of arms based on the provided configuration.

        Args:
            new_arms_config (dict): Dictionary containing the new configuration of arms.

        Returns:
            bool: True if the update was applied successfully, False otherwise.
        """
        new_arms = []
        for arm_config in new_arms_config.get('arms', []):
            if arm_config["name"] not in [arm.name for arm in self.bandit.base_arms]:
                new_arm = Arm(
                    name=arm_config['name'],
                    params=arm_config['params'],
                    sampler_type=SamplerType[arm_config['sampler_type']],
                    init_score=arm_config['score']
                )
                new_arms.append(new_arm)
        self.bandit.arms = new_arms
