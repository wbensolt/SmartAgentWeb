import json
import re
from typing import List, Dict

class CodeTravailLoader:
    def __init__(self, path: str = "data/code_du_travail.json"):
        with open(path, encoding="utf-8") as f:
            self.tree = json.load(f)
        self.articles = self._extract_articles(self.tree)

    def _extract_articles(self, node) -> List[Dict]:
        results = []
        if node.get("type") == "article":
            data = node.get("data", {})
            etat = data.get("etat", "").upper()
            # Ne prendre que les articles en vigueur
            if etat == "VIGUEUR":
                # Prendre le texte complet s'il existe
                content = data.get("texte", "")
                if not content:
                    # Sinon concaténer les alinéas (si présents)
                    content = " ".join(
                        p.get("value", "") for p in node.get("children", [])
                        if p.get("type") == "alinea"
                    ).strip()

                num = data.get("num") or node.get("title") or ""
                article_id = data.get("id") or node.get("id")
                cid = data.get("cid")

                results.append({
                    "id": article_id,
                    "title": f"Article {num}".strip(),
                    "cid": cid,
                    "content": content
                })

        # Recurse on children nodes
        for child in node.get("children", []):
            results.extend(self._extract_articles(child))

        return results

    def _clean_keywords(self, keywords: List[str]) -> List[str]:
        stopwords = {"le", "la", "les", "de", "du", "des", "et", "à", "en", "un", "une", "pour", "avec", "dans", "sur", "par", "au", "aux", "ce", "cet", "cette", "est", "sont", "qui", "que", "qui", "où", "mais"}
        cleaned = []
        for kw in keywords:
            # enlever ponctuation et chiffres
            kw = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\-]", " ", kw.lower())
            # découper en mots
            mots = kw.split()
            # garder mots pas stopwords et > 2 lettres
            filtered = [m for m in mots if m not in stopwords and len(m) > 2]
            cleaned.extend(filtered)
        return cleaned

    def search_by_keywords(self, keywords: List[str], max_results: int = 5) -> List[Dict]:
        keywords_cleaned = self._clean_keywords(keywords)

        # Scoring articles par nombre de mots clés présents
        scored_articles = []

        for article in self.articles:
            content_lower = article["content"].lower()
            title_lower = article.get("title", "").lower()

            # Comptage mots clés présents dans contenu + titre
            matched_keywords = [kw for kw in keywords_cleaned if kw in content_lower or kw in title_lower]
            score = len(matched_keywords)

            if score > 0:
                scored_articles.append((score, article))

        # Trier par score décroissant
        scored_articles.sort(key=lambda x: x[0], reverse=True)

        # Retourner max_results meilleurs articles
        return [article for score, article in scored_articles[:max_results]]
