import asyncio
from tfl.client import Client as TFLClient
from tfl.api_token import ApiToken
import xkcd
from typing import Dict, Union, List
from gnews import GNews
import python_weather
import random
import numpy as np
from typing import List
from enum import Enum


class Sampler:
    def __init__(self):
        raise NotImplementedError()

    async def sample(self, params: Dict[str, Union[str, float]] = {}):
        """
        Sample function that processes the input parameter string and interacts with the API.

        Parameters:
        params (Dict): Dict with arguments

        Returns:
        dict: A dictionary containing the response from the API.
        """
        raise NotImplementedError()


class GNewsSampler(Sampler):
    def __init__(self, max_results=1):
        # Instantiate the Google News client once
        self.google_news = GNews(max_results=max_results)

    async def sample(self, params: Dict[str, Union[str, float]] = {'query': 'World News'}):
        """
        Get news articles based on the 'query' parameter from the params dictionary.
        """
        query = params.get('query', '')
        news = self.google_news.get_news(query)
        if news:
            return news[0]  # Return the first news article
        else:
            return {"error": "No news articles found"}


class XKCDSampler(Sampler):
    def __init__(self):
        pass  # No need for an API key

    async def sample(self, params: Dict[str, Union[str, float]] = {}):
        """
        Get a random XKCD comic. The 'params' dictionary is ignored as no parameters are required.
        """
        comic = xkcd.getRandomComic()
        return vars(comic)  # Return the comic data as a dictionary


class WeatherSampler(Sampler):
    def __init__(self):
        # Create the Python Weather client once
        self.client = python_weather.Client(unit=python_weather.METRIC)

    async def sample(self, params: Dict[str, Union[str, float]] = {'location': 'London'}):
        """
        Fetch weather data for the given location from the 'location' parameter in the params dictionary.
        """
        location = params.get('location', '')
        weather = await self.client.get(location)
        # Assuming daily_forecasts is a generator, get the next 3 days
        daily = weather.daily_forecasts
        weather_report = "Weather Forecast\n"
        # get the weather forecast for a few days
        for daily in weather:
            weather_report += f"{daily}\n"
            # hourly forecasts
            for hourly in daily:
                weather_report += f" --> {hourly!r}"
        return weather_report

    async def close(self):
        """
        Close the client properly (if necessary). Use this when you're done with the instance.
        """
        await self.client.close()

class Arm:
    def __init__(self, name:str, params: Dict[str, Union[str,float]], sampler_type: Sampler, init_score=5, decay_rate=0.1):
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
    def __init__(self, arms: List[Arm], alpha: float):
        """
        Initializes the Bandit algorithm with a set of arms using exponential sampling based on score.

        Args:
            arms (List[Arm]): List of Arm objects.
            alpha (float): Exponential scaling factor for sampling probabilities.
        """
        self.arms = arms  # A list of Arm objects
        self.alpha = alpha  # Scaling factor for exponential distribution

    def get_probabilities(self) -> List[float]:
        """Calculates the probabilities of the arms based on their scores following an exponential distribution."""
        valid_scores = [arm.score**self.alpha for arm in self.arms if arm.score >= 4]
        
        if not valid_scores:
            raise ValueError("No arms with score >= 4 to sample from.")
        
        total_score = sum(valid_scores)
        probabilities = [score/total_score for score in valid_scores]
        return probabilities

    def select_arm(self) -> Arm:
        """Selects and returns the arm to pull based on the exponential distribution of the scores."""
        # Filter out arms with scores less than 4
        valid_arms = [arm for arm in self.arms if arm.score >= 4]
        
        if not valid_arms:
            raise ValueError("No valid arms with score >= 4 available for selection.")

        probabilities = self.get_probabilities()
        selected_arm = random.choices(valid_arms, weights=probabilities, k=1)[0]
        return selected_arm

    def pull_and_decay(self, arm: Arm):
        """Pulls the arm and applies the decay towards 5."""
        arm.sample_arm()

class SamplerType(Enum):
    GNEWS = 1
    XKCD = 2
    WEATHER = 3

class Item:
    def __init__(self, arm: Arm, sample_result: dict):
        self.arm = arm
        self.sample_result = sample_result

    def __repr__(self):
        return f"Item(arm={self.arm.name}, score={self.arm.score}, sample_result={self.sample_result})"


class BanditManager:
    def __init__(self, bandit: Bandit, samplers: dict):
        self.bandit = bandit
        self.samplers = samplers

    async def sample(self):
        # Select an arm using the bandit algorithm
        selected_arm = self.bandit.select_arm()
        selected_arm.sample_arm()

        # Get the corresponding sampler
        sampler_type = selected_arm.sampler_type
        sampler = self.samplers[sampler_type]

        sample = await sampler.sample(selected_arm.params)

        return Item(selected_arm, sample)