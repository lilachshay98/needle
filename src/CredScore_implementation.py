# -*- coding: utf-8 -*-

import csv
import logging
import math
import os
import string
import sys
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm
import warnings
import re
import networkx as nx

import pandas as pd
from joblib import load

from colorama import Fore, Style

from community_detection import load_accounts, build_follow_graph, top_mentions, \
    subgraph_around_anchors, louvain_partition, analyze_communities, predict_account_label
from page_rank import extract_domain, build_graph_from_edges, scrape_outlinks_one

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATS_DIR = os.path.join(BASE_DIR, 'data/stats')


class CredScore:
    """
    Comprehensive credibility assessment system for news articles and social media content.

    Integrates multiple machine learning models and external signals to provide
    credibility scores for news articles and bot detection for social media accounts.

    The system uses:
    - Ensemble of 4 ML models for text classification
    - Domain reputation scoring based on historical data
    - PageRank analysis for URL credibility
    - Temporal patterns in fake news distribution
    - Social network community analysis for bot detection

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Text vectorizer for converting articles to feature vectors
    news_models : dict
        Dictionary containing trained ML models for news classification
    bot_model : RandomForestClassifier
        Trained model for bot detection
    news_decision_threshold : float
        Threshold for binary classification decisions (default: 0.5)

    Methods
    -------
    predict_news(text, domain, date, url=None)
        Analyze news article credibility
    predict_bot(tweet_text, user_data)
        Detect if social media account is automated
    find_optimal_threshold(validation_texts, validation_labels, ...)
        Optimize decision threshold for balanced predictions
    get_url_pagerank_score(user_url, graph=None, alpha=0.85)
        Calculate PageRank-based credibility score for URLs
    """

    def __init__(self):
        """
        Initialize the CredScore classifier by loading all required models and data.

        Loads text vectorizer, news classification models (Naive Bayes, Logistic Regression,
        Decision Tree, Random Forest), and bot detection model. Displays colored status
        messages and logs all operations.

        Raises
        ------
        SystemExit
            If critical models cannot be loaded, exits with code 1
        FileNotFoundError
            If model files are missing from expected directories
        """
        print(f"{Fore.CYAN}Starting classification application...{Style.RESET_ALL}")
        logging.info("Starting classification application...")

        try:
            # Load vectorizer for news classification
            self.vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
            print(f"{Fore.CYAN}Loading vectorizer...{Style.RESET_ALL}")
            logging.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = load(self.vectorizer_path)

            self.news_decision_threshold = 0.5  # setting as starting default value which will be calibrated

            # Load news classification models
            self.news_models = {}
            model_files = {
                'naive_bayes': 'naive_bayes_model.joblib',
                'logistic_regression': 'logistic_regression_model.joblib',
                'decision_tree': 'decision_tree_model.joblib',
                'random_forest': 'random_forest_model.joblib'
            }

            print(f"{Fore.CYAN}Loading news classification models...{Style.RESET_ALL}")
            for name, filename in model_files.items():
                model_path = os.path.join(MODELS_DIR, filename)
                logging.info(f"Loading {name} model from {model_path}")
                self.news_models[name] = load(model_path)

            # Load bot detection model
            print(f"{Fore.CYAN}Loading bot detection model...{Style.RESET_ALL}")
            bot_model_path = os.path.join(MODELS_DIR, 'bots', 'random_forest_bot_detector_latest.joblib')
            if os.path.exists(bot_model_path):
                self.bot_model = load(bot_model_path)
                logging.info(f"Loaded bot detection model from {bot_model_path}")
            else:
                bot_models = [f for f in os.listdir(os.path.join(MODELS_DIR, 'bots')) if f.endswith('.joblib')]
                if bot_models:
                    latest_model = max(bot_models)
                    bot_model_path = os.path.join(MODELS_DIR, 'bots', latest_model)
                    self.bot_model = load(bot_model_path)
                    logging.info(f"Loaded bot detection model from {bot_model_path}")
                else:
                    self.bot_model = None
                    logging.warning("No bot detection model found")

            print(f"{Fore.GREEN}All models loaded successfully!{Style.RESET_ALL}")
            logging.info("All models loaded successfully")

        except Exception as e:
            print(f"{Fore.RED}Error loading models: {str(e)}{Style.RESET_ALL}")
            logging.error(f"Error loading models: {str(e)}")
            sys.exit(1)

    def set_threshold(self, threshold):
        """
        Set the decision threshold for binary classification predictions.

        Parameters
        ----------
        threshold : float
            Decision threshold between 0.0 and 1.0. Values above this threshold
            are classified as 'REAL', values below as 'FAKE'
        """
        self.news_decision_threshold = threshold

    def find_optimal_threshold(self, validation_texts, validation_labels, validation_domains=None,
                               validation_dates=None, validation_urls=None):
        """
        Find optimal decision threshold for balanced predictions using validation data.

        Tests thresholds from 0.35 to 0.65 in steps of 0.02, optimizing for F1-score.
        Uses tqdm progress bars to show optimization progress.

        Parameters
        ----------
        validation_texts : list of str
            List of news article texts for validation
        validation_labels : list of int
            Ground truth labels (0 for fake, 1 for real)
        validation_domains : list of str, optional
            Domain names for each article
        validation_dates : list of str, optional
            Publication dates in YYYY-MM format
        validation_urls : list of str, optional
            Full URLs for PageRank analysis

        Returns
        -------
        float
            Optimal threshold value that maximizes F1-score
        """
        thresholds = np.arange(0.40, 0.52, 0.02)
        best_threshold = 0.5
        best_f1 = 0

        threshold_pbar = tqdm(thresholds, desc="Testing thresholds", unit="threshold")

        for threshold in threshold_pbar:
            predictions = []

            # Inner progress bar for predictions at each threshold
            prediction_pbar = tqdm(range(len(validation_texts)),
                                   desc=f"Threshold {threshold:.2f}",
                                   unit="predictions",
                                   leave=False)

            for i in prediction_pbar:
                text = validation_texts[i]
                domain = validation_domains[i] if validation_domains else None
                date = validation_dates[i] if validation_dates else ""
                url = validation_urls[i] if validation_urls else None

                result = self.predict_news(text, domain, date, url)
                if result:
                    # Use threshold for prediction
                    pred = 1 if result['real_probability'] > (threshold * 100) else 0
                    predictions.append(pred)
                else:
                    predictions.append(0)

            # Calculate F1 score
            if len(predictions) == len(validation_labels):
                f1 = f1_score(validation_labels, predictions)

                # Update threshold progress bar with current metrics
                threshold_pbar.set_postfix({
                    'Current F1': f'{f1:.4f}',
                    'Best F1': f'{best_f1:.4f}',
                    'Best Threshold': f'{best_threshold:.2f}'
                })

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        threshold_pbar.close()
        return best_threshold

    def predict_news(self, text, domain, date, url=None):
        """
        Analyze news article credibility using ensemble of models and external signals.

        Combines predictions from multiple ML models with domain reputation,
        PageRank scores, and temporal patterns to produce final credibility assessment.

        Scoring weights:
        - Model prediction: 20%
        - Domain reputation: 40%
        - PageRank score: 40%

        Parameters
        ----------
        text : str
            News article text content
        domain : str
            Domain name of the news source
        date : str
            Publication date in YYYY-MM format
        url : str, optional
            Full URL for PageRank analysis. If provided but domain is None,
            domain will be extracted from URL

        Returns
        -------
        dict or None
            Dictionary containing:
            - prediction : int (0 for fake, 1 for real)
            - label : str ('FAKE' or 'REAL')
            - confidence : float (0-100, confidence in prediction)
            - real_probability : float (0-100, probability of being real)
            - fake_probability : float (0-100, probability of being fake)
            - model_votes : dict (individual model predictions)
            - score_contributions : dict (contribution from each component)
            - component_weights : dict (weight of each scoring component)

            Returns None if prediction fails due to errors
        """
        domain_stats = self.get_domain_stats()
        date_stats = self.get_date_stats()
        year, month = self.get_year_and_month_from_date_input(date)
        try:
            print(f"\n{Fore.CYAN}Analyzing text...{Style.RESET_ALL}")

            # Clean the text
            cleaned_text = self.clean_text(text)
            logging.info("Cleaning input text...")

            # Vectorize
            logging.info("Vectorizing text...")
            X = self.vectorizer.transform([cleaned_text])

            # Make predictions with each model
            results = {}
            probabilities = {}

            for name, model in self.news_models.items():
                logging.info(f"Getting prediction from {name}...")

                # Get prediction
                prediction = model.predict(X)[0]
                results[name] = prediction

                # Get probability from the model
                proba = model.predict_proba(X)[0]
                probabilities[name] = proba

            # Calculate voting result
            votes = list(results.values())
            model_score = sum(votes) / len(votes)  # Score between 0 and 1, hard voting score

            # Calculate average probabilities from models
            avg_proba = [0, 0]
            for name in probabilities:
                avg_proba[0] += probabilities[name][0]
                avg_proba[1] += probabilities[name][1]

            avg_proba[0] /= len(probabilities)
            avg_proba[1] /= len(probabilities)

            # Use avg_proba[1] as the model's real probability score
            model_real_prob = avg_proba[1]  # soft voting score

            # Weight by model confidence
            confidence_weighted_prob = 0
            total_confidence = 0

            for name, proba in probabilities.items():
                confidence = max(proba)
                confidence_weighted_prob += proba[1] * confidence
                total_confidence += confidence

            if total_confidence > 0:
                model_real_prob = confidence_weighted_prob / total_confidence

            # Combine with hard voting
            combined_model_score = (model_score * 0.5) + (model_real_prob * 0.5)

            # Initialize domain and page rank scores with neutral values
            domain_score = 0.5
            page_rank_score = 0.5

            # Track contribution from each source for reporting
            score_contributions = {
                'model_prediction': (combined_model_score - 0.5) * 0.2
            }

            # Extract domain from URL if provided but domain is not
            if url and not domain:
                domain = extract_domain(url)
                if domain:
                    print(f"{Fore.CYAN}Extracted domain from URL: {domain}{Style.RESET_ALL}")
                    logging.info(f"Domain extracted from URL: {domain}")

            # Apply domain reputation adjustment if available
            if domain and domain in domain_stats:
                logging.info(f"Adjusting scores based on domain: {domain}")
                fake_ratio = domain_stats[domain]
                # Convert fake_ratio to a credibility score (1 - fake_ratio)
                domain_score = 1.0 - fake_ratio
                score_contributions['domain_score'] = (domain_score - 0.5) * 0.4
                logging.info(f"Applied domain score: {domain_score}")

            # Get page rank and use it as a factor (higher page rank = more likely real)
            if url:
                pr_score, pr_message = self.get_url_pagerank_score(url)
                if pr_score is not None:
                    page_rank_score = pr_score
                    score_contributions['page_rank_score'] = (page_rank_score - 0.5) * 0.4
                    logging.info(f"Applied page rank score: {page_rank_score}, {pr_message}")

            # Apply date-based adjustment if available
            if year and month and year in date_stats and month in date_stats[year]:
                logging.info(f"Adjusting scores based on date: {date}")
                fake_ratio = date_stats[year][month]
                # Apply a small date adjustment to the model score
                date_adjustment = min(0.1, fake_ratio * 0.2)  # Cap the adjustment
                model_real_prob -= date_adjustment
                score_contributions['date_factor'] = -date_adjustment
                print(f"{Fore.YELLOW}Applied date-based adjustment: -{date_adjustment:.4f}{Style.RESET_ALL}")

            # Calculate final probability based on weighted components:
            # 20% model prediction, 40% domain score, 40% page rank
            final_real_probability = (
                    (combined_model_score * 0.2) +  # 20% model prediction
                    (domain_score * 0.4) +  # 40% domain score
                    (page_rank_score * 0.4)  # 40% page rank score
            )

            # Ensure probability is within bounds
            final_real_probability = max(0.01, min(0.99, final_real_probability))
            final_fake_probability = 1.0 - final_real_probability

            # Make final prediction
            final_prediction = 1 if final_real_probability > self.news_decision_threshold else 0

            logging.info(f"Final probability - Real: {final_real_probability:.4f}, Fake: {final_fake_probability:.4f}")
            logging.info(f"Score contributions: {score_contributions}")
            logging.info(f"Component weights - Model: 40%, Domain: 30%, PageRank: 30%")

            return {
                'prediction': final_prediction,
                'label': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': max(final_real_probability, final_fake_probability) * 100,
                'real_probability': final_real_probability * 100,
                'fake_probability': final_fake_probability * 100,
                'model_votes': results,
                'score_contributions': score_contributions,
                'component_weights': {
                    'model_prediction': '40%',
                    'domain_score': '30%',
                    'page_rank_score': '30%'
                }
            }

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

    def predict_bot(self, tweet_text, user_data):
        """
        Analyze if a social media account is automated (bot) based on profile and content.

        Uses ensemble approach combining:
        - Random Forest model trained on user profile features
        - Rule-based indicators from profile completeness analysis
        - Tweet content analysis (hashtags, mentions, URLs)
        - Community detection scores

        Strong human indicators (with AUC scores):
        - Verified status (0.7828)
        - Followers/friends ratio (0.7501)
        - Follower count (0.7357)
        - Listed count (0.7338)
        - Account age (0.6287)

        Parameters
        ----------
        tweet_text : str
            Text content of the tweet to analyze
        user_data : dict
            Dictionary containing Twitter user metrics:
            - followers_count : int
            - friends_count : int
            - verified : bool
            - created_at : str (Twitter date format)
            - statuses_count : int
            - favourites_count : int
            - listed_count : int
            - screen_name : str

        Returns
        -------
        dict or None
            Dictionary containing:
            - prediction : int (0 for bot, 1 for human)
            - label : str ('HUMAN', 'BOT', 'UNKNOWN', or 'ERROR')
            - confidence : float (0-100, confidence in prediction)
            - bot_probability : float (0-100, probability of being bot)
            - human_probability : float (0-100, probability of being human)
            - human_indicators : dict (scores from profile analysis)
            - tweet_features : dict (extracted tweet characteristics)
            - community_score : dict (community detection results)

            Returns None if analysis fails due to errors
        """
        try:
            if self.bot_model is None:
                logging.error("Bot detection model not available")
                return {
                    'prediction': None,
                    'label': 'UNKNOWN',
                    'message': 'Bot detection model not available',
                    'confidence': 0
                }

            print(f"\n{Fore.CYAN}Analyzing account for bot characteristics...{Style.RESET_ALL}")

            # Apply the 5 strong human likelihood indicators based on profile_completeness_auc.csv
            # These indicators have high AUC values indicating strong discriminatory power
            human_score = 0.0
            human_indicators = {}

            # 1. Verified status (binary) - AUC: 0.7828
            verified_weight = 0.7828
            if 'verified' in user_data and user_data['verified']:
                human_indicators['verified'] = verified_weight
                human_score += verified_weight

            # 2. Followers to friends ratio - AUC: 0.7501
            followers_to_friends_ratio_weight = 0.7501
            if 'followers_count' in user_data and 'friends_count' in user_data:
                followers_to_friends_ratio = user_data['followers_count'] / (user_data['friends_count'] + 1)
                # Normalize the ratio: most human accounts have ratios between 0.1 and 10
                normalized_ratio = min(1.0, followers_to_friends_ratio / 10.0)
                human_indicators['followers_to_friends_ratio'] = normalized_ratio * followers_to_friends_ratio_weight
                human_score += human_indicators['followers_to_friends_ratio']

            # 3. Number of followers - AUC: 0.7357
            followers_weight = 0.7357
            if 'followers_count' in user_data:
                # Normalize followers count: logarithmic scale (1000 followers is a significant threshold)
                followers_normalized = min(1.0, math.log10(user_data['followers_count'] + 1) / 3.0)
                human_indicators['followers'] = followers_normalized * followers_weight
                human_score += human_indicators['followers']

            # 4. Listed count - AUC: 0.7338
            listed_count_weight = 0.7338
            if 'listed_count' in user_data:
                # Normalize listed count: being in 10+ lists is significant
                listed_normalized = min(1.0, user_data['listed_count'] / 10.0)
                human_indicators['listed_count'] = listed_normalized * listed_count_weight
                human_score += human_indicators['listed_count']

            # 5. Account age in days - AUC: 0.6287
            account_age_weight = 0.6287
            if 'created_at' in user_data:
                try:
                    # Parse Twitter date format
                    created_date = datetime.strptime(user_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    current_date = datetime.now()
                    account_age_days = (current_date - created_date).days
                    # Normalize account age: accounts older than 365 days get full score
                    age_normalized = min(1.0, account_age_days / 365.0)
                    human_indicators['account_age_days'] = age_normalized * account_age_weight
                    human_score += human_indicators['account_age_days']
                except Exception as e:
                    logging.error(f"Error calculating account age: {str(e)}")

            # Calculate human probability based on indicators
            # Maximum possible score if all indicators are at their max
            max_possible_score = verified_weight + followers_to_friends_ratio_weight + followers_weight + \
                                 listed_count_weight + account_age_weight
            # Normalize to a probability
            human_probability = min(0.95, human_score / max_possible_score)

            # Extract features from tweet text
            tweet_features = {}
            if tweet_text:
                # Number of hashtags
                tweet_features['hashtag'] = len(re.findall(r'#\w+', tweet_text))

                # Number of mentions
                tweet_features['mentions'] = len(re.findall(r'@\w+', tweet_text))

                # Number of URLs
                tweet_features['uniqueURL'] = len(
                    re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                               tweet_text))

                # Unique hashtags and mentions
                tweet_features['uniqueHashtags'] = len(set(re.findall(r'#\w+', tweet_text)))
                tweet_features['uniqueMentions'] = len(set(re.findall(r'@\w+', tweet_text)))

            # Prepare features for bot detection using the existing model
            features = {}

            # Required features based on the bot model
            required_features = [
                'followers_count', 'friends_count', 'statuses_count',
                'favourites_count', 'listed_count', 'screen_name_length',
                'retweets', 'replies', 'favoriteC', 'hashtag',
                'mentions', 'intertime', 'ffratio', 'favorites',
                'uniqueHashtags', 'uniqueMentions', 'uniqueURL'
            ]

            # Fill available features from user_data
            for feature in required_features:
                if feature in user_data:
                    features[feature] = user_data[feature]
                elif feature in tweet_features:
                    features[feature] = tweet_features[feature]
                else:
                    # Use sensible defaults for missing features
                    if feature == 'ffratio' and 'followers_count' in user_data and 'friends_count' in user_data:
                        if user_data['followers_count'] > 0:
                            features[feature] = user_data['friends_count'] / user_data['followers_count']
                        else:
                            features[feature] = user_data['friends_count'] if user_data['friends_count'] > 0 else 1.0
                    elif feature == 'screen_name_length' and 'screen_name' in user_data:
                        features[feature] = len(user_data['screen_name'])
                    else:
                        # Default values for other features
                        features[feature] = 0.0

            # Convert to DataFrame with appropriate columns that the model expects
            X = pd.DataFrame([features])

            # Add URL field which was in the original dataset
            if 'url' not in X.columns:
                X['url'] = 0.0

            # Add 'listed' field which appears in the feature importance file
            if 'listed' not in X.columns:
                X['listed'] = X['listed_count'] if 'listed_count' in X.columns else 0.0

            # Add polynomial feature interactions - explicitly listing all needed combinations
            # to ensure we have exactly what the model expects
            interaction_pairs = [
                ('screen_name_length', 'statuses_count'),
                ('followers_count', 'friends_count'),
                ('screen_name_length', 'friends_count'),
                ('screen_name_length', 'followers_count'),
                ('followers_count', 'favourites_count'),
                ('friends_count', 'favourites_count'),
                ('screen_name_length', 'favourites_count'),
                ('statuses_count', 'followers_count'),
                ('statuses_count', 'friends_count'),
                ('statuses_count', 'favourites_count')
            ]

            for feat1, feat2 in interaction_pairs:
                interaction_name = f"{feat1} {feat2}"
                X[interaction_name] = X[feat1] * X[feat2]

            # Verify we have all 29 features
            expected_features = {'screen_name_length', 'statuses_count', 'followers_count', 'friends_count',
                                 'favourites_count', 'listed_count', 'url', 'retweets', 'replies', 'favoriteC',
                                 'hashtag', 'mentions', 'intertime', 'ffratio', 'favorites', 'uniqueHashtags',
                                 'uniqueMentions', 'uniqueURL', 'listed', 'screen_name_length statuses_count',
                                 'followers_count friends_count', 'screen_name_length friends_count',
                                 'screen_name_length followers_count', 'followers_count favourites_count',
                                 'friends_count favourites_count', 'screen_name_length favourites_count',
                                 'statuses_count followers_count', 'statuses_count friends_count',
                                 'statuses_count favourites_count'}

            # Check if we're missing any features and add them
            missing_features = expected_features - set(X.columns)
            for feat in missing_features:
                X[feat] = 0.0  # Add any missing features with default value
                logging.info(f"Added missing feature: {feat}")

            # Make prediction
            try:
                # Try to extract feature names from the model
                model_features = []
                if hasattr(self.bot_model, 'feature_names_in_'):
                    model_features = self.bot_model.feature_names_in_.tolist()

                # If model has feature names, ensure we have exactly those features
                if model_features:
                    # Add any missing features
                    for feat in model_features:
                        if feat not in X.columns:
                            X[feat] = 0.0
                    # Keep only the features the model knows about and in the same order
                    X = X[model_features]

                # Suppress the specific warning about feature names
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning,
                                            message="X has feature names, but RandomForestClassifier was fitted "
                                                    "without feature names")

                    # make the prediction
                    prediction = self.bot_model.predict(X)[0]
                    probabilities = self.bot_model.predict_proba(X)[0]

                # Bot is labeled as 0, human as 1
                # Blend model prediction with our human indicators
                model_human_prob = probabilities[1]

                # Add tweet text features to the human indicators
                tweet_human_score = 0.0
                if tweet_text:
                    # Bot accounts often have more hashtags, mentions, and URLs than human accounts
                    # Higher hashtag count suggests potential bot activity
                    hashtag_weight = 0.05
                    if tweet_features['hashtag'] > 3:
                        normalized_hashtags = max(0, 1 - (tweet_features['hashtag'] / 10))  # Penalize many hashtags
                        human_indicators['hashtag_count'] = normalized_hashtags * hashtag_weight
                        tweet_human_score += human_indicators['hashtag_count']

                    # Many mentions often indicate automated behavior
                    mention_weight = 0.05
                    if tweet_features['mentions'] > 2:
                        normalized_mentions = max(0, 1 - (tweet_features['mentions'] / 8))  # Penalize many mentions
                        human_indicators['mention_count'] = normalized_mentions * mention_weight
                        tweet_human_score += human_indicators['mention_count']

                    # Many URLs often indicate spam behavior
                    url_weight = 0.05
                    if tweet_features['uniqueURL'] > 0:
                        normalized_urls = max(0, 1 - (tweet_features['uniqueURL'] / 3))  # Penalize many URLs
                        human_indicators['url_count'] = normalized_urls * url_weight
                        tweet_human_score += human_indicators['url_count']

                # Add tweet text score with small weight
                text_weight = 0.15
                human_probability = human_probability * (1 - text_weight) + (
                        tweet_human_score / (hashtag_weight + mention_weight + url_weight)) * text_weight

                # Use a weighted average, giving more weight to our custom indicators for the specified features
                final_human_prob = (human_probability * 0.7) + (model_human_prob * 0.3)

                # Get community prediction if screen name is provided
                community_score = None
                try:
                    if user_data.get('screen_name'):
                        community_score = self.get_community_prediction_score(user_data['screen_name'])
                        # Ensure community_score is a number we can use
                        if isinstance(community_score, (int, float)):
                            # Incorporate community score into final human probability
                            final_human_prob = (final_human_prob * 0.7) + (community_score['human_probability'] * 0.3)
                        else:
                            logging.warning(f"Community score is not a number: {community_score}")
                except Exception as e:
                    logging.warning(f"Error getting community prediction: {str(e)}")

                # Determine if it's a bot based on final probability
                is_bot = final_human_prob < 0.5

                return {
                    'prediction': int(is_bot),
                    'label': 'BOT' if is_bot else 'HUMAN',
                    'confidence': (1 - final_human_prob if is_bot else final_human_prob) * 100,
                    'bot_probability': (1 - final_human_prob) * 100,
                    'human_probability': final_human_prob * 100,
                    'human_indicators': human_indicators,
                    'tweet_features': tweet_features,
                    'community_score': community_score
                }
            except Exception as e:
                logging.error(f"Error using bot prediction model: {str(e)}")
                logging.error(f"Current feature count: {len(X.columns)}, Features: {X.columns.tolist()}")
                return {
                    'prediction': None,
                    'label': 'ERROR',
                    'message': f"Error using model: {str(e)}",
                    'confidence': 0
                }

        except Exception as e:
            logging.error(f"Error in bot prediction: {str(e)}")
            return None

    @staticmethod
    def get_url_pagerank_score(user_url, graph=None, alpha=0.85):
        """
        Get PageRank score for a user-provided URL by:
        1. Extracting its domain
        2. Checking if it's a trusted platform
        3. If domain exists in graph, return its score
        4. If not, scrape its outlinks and calculate a temporary score

        Calculate PageRank score for a URL's domain with trusted platform handling.

        Computes PageRank-based credibility score by first checking
        existing graph data and scraping outlinks for new domains
        to calculate temporary scores.

        Parameters
        ----------
        user_url : str
            URL to analyze for PageRank scoring.
        graph : networkx.DiGraph, optional
            Existing domain graph for PageRank calculation. If None, loads
            from domain_edges.csv file.
        alpha : float, default=0.85
            Damping parameter for PageRank algorithm.

        Returns
        -------
        score : float
            PageRank score between 0.0 and 1.0, where higher values indicate
            more credible/authoritative domains.
        message : str
            Descriptive message explaining the score source and calculation.

        Notes
        -----
        Scoring hierarchy:
        1. Trusted platforms: Predefined high scores for major news outlets
        2. Existing graph: PageRank from historical domain relationship data
        3. Dynamic scraping: Temporary score from newly scraped outlinks
        4. Fallback: 0.5 neutral score for unreachable/invalid domains
        """
        # Extract domain from URL
        domain = extract_domain(user_url)
        if not domain:
            logging.warning(f"Could not extract a valid domain from the URL: {user_url}")
            return 0.5, "Could not extract a valid domain from the URL."

        # Load existing graph if not provided
        if graph is None:
            try:
                # Try to load existing edges
                edges = []
                with open(os.path.join(STATS_DIR, "domain_edges.csv"), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for src, dst, label in reader:
                        edges.append((src, dst, label))

                # Import networkx here to avoid dependency issues if not needed
                import networkx as nx
                graph = build_graph_from_edges(edges)
            except Exception as e:
                logging.error(f"Error loading existing graph: {str(e)}")
                return 0.5, f"Error loading existing graph: {str(e)}"

        try:
            # If domain already in graph, return its PageRank score
            pr = nx.pagerank(graph, alpha=alpha)
            if domain in pr:
                rank_position = sorted(pr.values(), reverse=True).index(pr[domain]) + 1
                logging.info(f"Domain {domain} exists in graph (rank {rank_position}/{len(pr)})")
                return pr[domain], f"Domain {domain} exists in our database (rank {rank_position}/{len(pr)})"

            # If domain not in graph, fetch its outlinks and calculate temporary score
            logging.info(f"Domain {domain} not in existing graph. Fetching outlinks...")

            # Get outlinks for this domain
            outlinks = scrape_outlinks_one(user_url)
            if not outlinks:
                logging.warning(f"Could not fetch any outlinks for {domain}")
                return 0.5, f"Could not fetch any outlinks for {domain}"

            # Create temporary graph with new domain and its connections
            temp_graph = graph.copy()
            for src, dst in outlinks:
                temp_graph.add_edge(src, dst)

            # Calculate new PageRank scores
            new_pr = nx.pagerank(temp_graph, alpha=alpha)

            # Return the score for our domain
            if domain in new_pr:
                rank_position = sorted(new_pr.values(), reverse=True).index(new_pr[domain]) + 1
                logging.info(f"Calculated temporary score for {domain} (rank {rank_position}/{len(new_pr)})")
                return new_pr[domain], f"Temporary score for {domain} (rank {rank_position}/{len(new_pr)})"
            else:
                logging.warning(f"Domain {domain} has no connections in the graph")
                return 0.5, f"Domain {domain} has no connections in the graph"

        except Exception as e:
            logging.error(f"Error calculating PageRank: {str(e)}")
            return 0.5, f"Error calculating PageRank: {str(e)}"

    @staticmethod
    def get_community_prediction_score(account_name):
        """
        Analyze account credibility using social network community detection.

        Loads Twitter follow graph, identifies influential anchors, creates subgraph,
        detects communities using Louvain algorithm, and predicts account label
        based on community characteristics.

        Parameters
        ----------
        account_name : str
            Twitter screen name (without @) to analyze

        Returns
        -------
        float or dict
            Community-based credibility score or prediction result.
            Returns 1.0 (neutral) if account_name is not provided.
        """

        if not account_name:
            return 1.0  # Neutral score if no account name provided
        # Load and build graph
        accounts = load_accounts()
        Gd, labels, screen = build_follow_graph(accounts)

        # Find anchors and create subgraph
        anchors = top_mentions(accounts)
        H, anchor_ids = subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, mutual_only=False)

        # Detect communities
        partition = louvain_partition(H)

        # Analyze communities
        community_stats = analyze_communities(H, labels, partition)

        return predict_account_label(account_name, Gd, H, labels, screen, partition, community_stats, radius=2)

    @staticmethod
    def get_domain_stats():
        """
        Load domain reputation statistics from CSV file.

        Reads domains_summary.csv containing historical fake news ratios
        for different domains to inform credibility scoring.

        Returns
        -------
        dict
            Dictionary mapping domain names (str) to fake ratios (float)
            where 0.0 = never fake, 1.0 = always fake
        """
        domain_stats = {}
        domains_path = os.path.join(STATS_DIR, 'domains_summary.csv')
        if os.path.exists(domains_path):
            with open(domains_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    domain = parts[0].strip().lower()
                    if domain == '﻿domain':
                        continue
                    fake_ratio = float(parts[4].strip())
                    domain_stats[domain] = fake_ratio
        else:
            logging.warning(f"domains.txt file not found at {domains_path}")
        return domain_stats

    @staticmethod
    def get_date_stats():
        """
        Load temporal fake news statistics from CSV file.

        Reads monthly_bot_data.csv containing fake news ratios by
        year and month to identify temporal patterns.

        Returns
        -------
        dict
            Nested dictionary with structure:
            {year: {month: fake_ratio}}
            where fake_ratio is float between 0.0 and 1.0
        """
        dates_stats = {}
        dates_path = os.path.join(STATS_DIR, 'monthly_bot_data.csv')
        if os.path.exists(dates_path):
            with open(dates_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    year = parts[5].strip()
                    if year == 'year':
                        continue  # Skip header
                    if not dates_stats.get(year, None):
                        dates_stats[year] = {}
                    month = parts[6].strip()
                    fake_ratio = float(parts[3].strip())
                    dates_stats[year][month] = fake_ratio
        return dates_stats

    @staticmethod
    def clean_text(text):
        """
        Preprocess text using same cleaning pipeline as training data.

        Applies lowercase conversion, punctuation removal, and whitespace
        normalization to ensure consistency with model training.

        Parameters
        ----------
        text : str or any
            Input text to clean. Non-string inputs converted to string.

        Returns
        -------
        str
            Cleaned text ready for vectorization
        """
        logging.info("Cleaning input text...")

        if not isinstance(text, str):
            text = str(text)
        # lowercase text
        text = text.lower()
        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # remove extra whitespace
        text = ' '.join(text.split())

        return text

    @staticmethod
    def get_year_and_month_from_date_input(date):
        """
        Parse date string into year and month components.

        Validates date format and extracts components for temporal analysis.
        Expected format: "YYYY-MM" (e.g., "2024-01", "2024-12")

        Parameters
        ----------
        date : str
            Date string in YYYY-MM format

        Returns
        -------
        tuple of (str, str) or (None, None)
            Year and month as strings, or (None, None) if parsing fails.
            Month is normalized to remove leading zeros
        """
        if date and '-' in date:
            parts = date.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                year, month = parts
                if len(month) == 2 and month.startswith('0'):
                    month = month[1:]
                return year, month
        return None, None
