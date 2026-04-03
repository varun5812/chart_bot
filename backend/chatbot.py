from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class KnowledgeEntry:
    intent: str
    answer: str
    training_phrases: List[str]


class CareerChatbot:
    """Simple NLP chatbot powered by TF-IDF sentence matching."""

    def __init__(self) -> None:
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

    def get_response(self, message: str) -> str:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        query_vector = self.vectorizer.transform([cleaned_message])
        similarity_scores = cosine_similarity(query_vector, self.training_matrix).flatten()

        best_index = int(np.argmax(similarity_scores))
        best_score = float(similarity_scores[best_index])

        if best_score < 0.18:
            return self._fallback_response()

        matched_intent = self.intent_lookup[best_index]
        return self.intent_to_answer[matched_intent]

    @staticmethod
    def _fallback_response() -> str:
        return (
            "I can help with data science careers, skill building, interview prep, and "
            "learning roadmaps. Try asking about Python, SQL, machine learning, projects, "
            "resume tips, or how to become a data scientist."
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
