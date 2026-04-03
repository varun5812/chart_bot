from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
WEB_SEARCH_TRIGGER_WORDS = {
    "latest",
    "recent",
    "current",
    "today",
    "news",
    "trend",
    "trends",
    "salary",
    "salaries",
    "market",
    "hiring",
    "jobs",
}


@dataclass(frozen=True)
class KnowledgeEntry:
    intent: str
    answer: str
    training_phrases: List[str]


@dataclass(frozen=True)
class SearchResult:
    title: str
    link: str
    snippet: str


@dataclass(frozen=True)
class ChatbotReply:
    response: str
    sources: List[SearchResult] = field(default_factory=list)
    mode: str = "knowledge-base"


class GoogleSearchClient:
    """Thin wrapper around the official Google Custom Search JSON API."""

    def __init__(self, api_key: str | None = None, search_engine_id: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_CSE_ID")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.search_engine_id)

    def search(self, query: str, num_results: int = 3) -> List[SearchResult]:
        if not self.enabled:
            return []

        params = urlencode(
            {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": max(1, min(num_results, 10)),
            }
        )
        request_url = f"{GOOGLE_SEARCH_URL}?{params}"

        try:
            with urlopen(request_url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []

        items = payload.get("items", [])
        return [
            SearchResult(
                title=item.get("title", "Untitled result"),
                link=item.get("link", ""),
                snippet=item.get("snippet", "").replace("\n", " ").strip(),
            )
            for item in items
            if item.get("link")
        ]


class CareerChatbot:
    """Simple NLP chatbot with an optional Google search fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        search_engine_id: str | None = None,
    ) -> None:
        self.knowledge_base = self._build_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.training_phrases = [
            phrase
            for entry in self.knowledge_base
            for phrase in entry.training_phrases
        ]
        self.intent_lookup = [
            entry.intent
            for entry in self.knowledge_base
            for _ in entry.training_phrases
        ]
        self.training_matrix = self.vectorizer.fit_transform(self.training_phrases)
        self.intent_to_answer = {
            entry.intent: entry.answer for entry in self.knowledge_base
        }
        self.intent_catalog = pd.DataFrame(
            [{"intent": entry.intent, "answer": entry.answer} for entry in self.knowledge_base]
        )
        self.google_search = GoogleSearchClient(api_key, search_engine_id)

    def get_response(self, message: str) -> ChatbotReply:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        matched_reply, best_score = self._match_knowledge_base(cleaned_message)

        if self._should_use_web_search(cleaned_message, best_score):
            web_results = self.google_search.search(cleaned_message)
            if web_results:
                return self._build_web_reply(cleaned_message, web_results)

        if matched_reply is not None:
            return ChatbotReply(response=matched_reply, mode="knowledge-base")

        return ChatbotReply(response=self._fallback_response(), mode="fallback")

    def _match_knowledge_base(self, message: str) -> tuple[str | None, float]:
        query_vector = self.vectorizer.transform([message])
        similarity_scores = cosine_similarity(query_vector, self.training_matrix).flatten()

        best_index = int(np.argmax(similarity_scores))
        best_score = float(similarity_scores[best_index])

        if best_score < 0.18:
            return None, best_score

        matched_intent = self.intent_lookup[best_index]
        return self.intent_to_answer[matched_intent], best_score

    def _should_use_web_search(self, message: str, similarity_score: float) -> bool:
        message_tokens = {token.lower().strip(".,?!") for token in message.split()}
        asks_for_fresh_info = any(token in WEB_SEARCH_TRIGGER_WORDS for token in message_tokens)
        return self.google_search.enabled and (similarity_score < 0.28 or asks_for_fresh_info)

    @staticmethod
    def _build_web_reply(query: str, results: List[SearchResult]) -> ChatbotReply:
        top_results = results[:3]
        snippets = [result.snippet for result in top_results if result.snippet]
        summary = " ".join(snippets)
        if not summary:
            summary = (
                f"I searched the web for '{query}' and found relevant sources. "
                "Open the links below for the latest details."
            )
        else:
            summary = (
                f"Here is a web-assisted answer for '{query}': {summary} "
                "Use the source links below for the latest details."
            )

        return ChatbotReply(response=summary, sources=top_results, mode="web-search")

    @staticmethod
    def _fallback_response() -> str:
        return (
            "I can help with data science careers, skill building, interview prep, learning "
            "roadmaps, and web-assisted answers when Google search is configured. Try asking "
            "about Python, SQL, machine learning, salaries, hiring trends, or how to become a data scientist."
        )

    @staticmethod
    def _build_knowledge_base() -> List[KnowledgeEntry]:
        return [
            KnowledgeEntry(
                intent="career_overview",
                answer=(
                    "A data science career usually blends statistics, programming, and business "
                    "problem-solving. Common roles include Data Analyst, Data Scientist, Machine "
                    "Learning Engineer, and Analytics Engineer. A strong starting path is to learn "
                    "Python, SQL, data visualization, statistics, and real-world project building."
                ),
                training_phrases=[
                    "What is a data science career",
                    "Tell me about data science jobs",
                    "How do I start a career in data science",
                    "What roles are there in data science",
                    "Career options in data science",
                ],
            ),
            KnowledgeEntry(
                intent="skills",
                answer=(
                    "Core skills for data science are Python, SQL, statistics, machine learning, "
                    "data cleaning, visualization, and communication. Tools such as pandas, NumPy, "
                    "scikit-learn, Jupyter, Git, and BI tools like Power BI or Tableau are also very useful."
                ),
                training_phrases=[
                    "What skills do I need for data science",
                    "Suggest data science skills",
                    "Important skills for data scientist",
                    "What should I learn for data science",
                    "Skills needed for machine learning jobs",
                ],
            ),
            KnowledgeEntry(
                intent="python_sql",
                answer=(
                    "Python and SQL are two of the most important skills. Python helps with data "
                    "analysis, machine learning, automation, and experimentation. SQL is essential "
                    "for querying data, joining tables, aggregating metrics, and answering business questions."
                ),
                training_phrases=[
                    "Why learn Python for data science",
                    "Is SQL important for data science",
                    "Should I learn Python and SQL",
                    "How important is SQL",
                    "Why is Python useful in data science",
                ],
            ),
            KnowledgeEntry(
                intent="roadmap",
                answer=(
                    "A practical roadmap is: 1) learn Python basics, pandas, and NumPy, 2) learn SQL "
                    "for querying data, 3) study statistics and probability, 4) build visualization "
                    "skills with matplotlib, seaborn, or Power BI, 5) learn machine learning with scikit-learn, "
                    "6) create portfolio projects, and 7) practice interviews and resume storytelling."
                ),
                training_phrases=[
                    "Give me a data science roadmap",
                    "How do I learn data science step by step",
                    "Learning roadmap for data science",
                    "Roadmap to become data scientist",
                    "Study plan for data science career",
                ],
            ),
            KnowledgeEntry(
                intent="projects",
                answer=(
                    "Good beginner-to-intermediate projects include sales forecasting, customer churn prediction, "
                    "movie recommendation systems, resume screening, sentiment analysis, dashboard creation, "
                    "and A/B testing analysis. Choose projects that show business impact, not just model accuracy."
                ),
                training_phrases=[
                    "What projects should I build",
                    "Data science project ideas",
                    "Portfolio projects for data science",
                    "Projects for machine learning resume",
                    "Best projects for beginners",
                ],
            ),
            KnowledgeEntry(
                intent="interview_prep",
                answer=(
                    "For interviews, prepare in four areas: SQL queries, Python problem-solving, statistics, "
                    "and machine learning concepts. Practice explaining trade-offs, data cleaning decisions, "
                    "evaluation metrics, and project impact. Be ready to discuss one or two portfolio projects in depth."
                ),
                training_phrases=[
                    "How do I prepare for data science interviews",
                    "Interview tips for data science",
                    "Data scientist interview preparation",
                    "What questions are asked in data science interview",
                    "Help me prepare for interviews",
                ],
            ),
            KnowledgeEntry(
                intent="resume",
                answer=(
                    "A strong data science resume should highlight measurable impact, technical skills, and projects. "
                    "Use bullet points like 'Built a churn model with 85 percent recall' or 'Created a dashboard that "
                    "reduced reporting time by 40 percent'. Keep projects relevant and include GitHub links when possible."
                ),
                training_phrases=[
                    "Resume tips for data science",
                    "How to make data science resume",
                    "What should I add in my resume",
                    "Resume advice for data scientist",
                    "How to improve my data science CV",
                ],
            ),
            KnowledgeEntry(
                intent="freshers",
                answer=(
                    "If you are a beginner, focus on fundamentals and consistency. Learn Python, SQL, statistics, "
                    "and one machine learning workflow. Build 2 to 4 polished projects, write short project summaries, "
                    "and apply for analyst, junior data scientist, internship, and business intelligence roles."
                ),
                training_phrases=[
                    "Can a fresher get into data science",
                    "How do beginners start data science",
                    "I am new to data science",
                    "Data science for freshers",
                    "Beginner advice for data science career",
                ],
            ),
            KnowledgeEntry(
                intent="learning_resources",
                answer=(
                    "A balanced learning path is to use Python tutorials, SQL practice platforms, statistics courses, "
                    "and project-based machine learning content. Combine learning with hands-on notebooks, GitHub portfolio work, "
                    "and short write-ups so your progress becomes visible to employers."
                ),
                training_phrases=[
                    "Best resources to learn data science",
                    "Where should I learn data science",
                    "Courses for data science",
                    "How to study data science online",
                    "Learning resources for machine learning",
                ],
            ),
            KnowledgeEntry(
                intent="salary_growth",
                answer=(
                    "Data science can offer strong growth because companies value people who turn data into decisions. "
                    "Salary depends on location, skills, domain knowledge, and experience. Improving SQL, Python, ML, "
                    "cloud basics, and communication often increases your chances of landing better roles."
                ),
                training_phrases=[
                    "Is data science a good career",
                    "Does data science pay well",
                    "Career growth in data science",
                    "Future of data science career",
                    "Is data science worth it",
                ],
            ),
        ]
