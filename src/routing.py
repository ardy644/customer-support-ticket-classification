"""Support ticket routing engine for BANKING77 classification."""
import numpy as np
from src.preprocessing import preprocess_text


# Maps department name -> list of intent categories
ROUTING_MAP = {
    'Card Services': [
        'card_arrival', 'card_delivery_estimate', 'card_about_to_expire',
        'card_acceptance', 'card_linking', 'card_not_working',
        'card_swallowed', 'contactless_not_working', 'virtual_card_not_working',
        'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card',
        'getting_virtual_card', 'order_physical_card', 'activate_my_card',
        'apple_pay_or_google_pay', 'visa_or_mastercard',
        'supported_cards_and_currencies', 'disposable_card_limits',
    ],
    'Card Security': [
        'lost_or_stolen_card', 'compromised_card', 'lost_or_stolen_phone',
        'pin_blocked', 'change_pin',
    ],
    'Payment Support': [
        'card_payment_fee_charged', 'card_payment_not_recognised',
        'card_payment_wrong_exchange_rate', 'declined_card_payment',
        'pending_card_payment', 'extra_charge_on_statement',
        'transaction_charged_twice', 'reverted_card_payment?',
        'direct_debit_payment_not_recognised',
    ],
    'Transfer Support': [
        'cancel_transfer', 'declined_transfer', 'failed_transfer',
        'pending_transfer', 'transfer_fee_charged', 'transfer_into_account',
        'transfer_not_received_by_recipient', 'transfer_timing',
        'receiving_money', 'beneficiary_not_allowed',
        'balance_not_updated_after_bank_transfer',
    ],
    'Cash & ATM Services': [
        'atm_support', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised',
        'declined_cash_withdrawal', 'pending_cash_withdrawal',
        'wrong_amount_of_cash_received',
        'balance_not_updated_after_cheque_or_cash_deposit',
    ],
    'Top-Up Support': [
        'automatic_top_up', 'pending_top_up', 'top_up_by_bank_transfer_charge',
        'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed',
        'top_up_limits', 'top_up_reverted', 'topping_up_by_card', 'verify_top_up',
    ],
    'Foreign Exchange': [
        'exchange_charge', 'exchange_rate', 'exchange_via_app',
        'fiat_currency_support', 'wrong_exchange_rate_for_cash_withdrawal',
        'country_support',
    ],
    'Account Management': [
        'edit_personal_details', 'terminate_account', 'age_limit',
        'passcode_forgotten',
    ],
    'Refunds': [
        'Refund_not_showing_up', 'request_refund',
    ],
    'Identity & Verification': [
        'unable_to_verify_identity', 'verify_my_identity',
        'verify_source_of_funds', 'why_verify_identity',
    ],
}

# Intent severity for priority assignment
HIGH_PRIORITY_INTENTS = [
    'lost_or_stolen_card', 'compromised_card', 'lost_or_stolen_phone',
    'declined_card_payment', 'failed_transfer', 'transaction_charged_twice',
    'pin_blocked',
]

MEDIUM_PRIORITY_INTENTS = [
    'card_not_working', 'contactless_not_working', 'virtual_card_not_working',
    'declined_cash_withdrawal', 'top_up_failed', 'declined_transfer',
    'unable_to_verify_identity',
]


def _build_reverse_map() -> dict:
    """Build intent -> department lookup."""
    reverse = {}
    for dept, intents in ROUTING_MAP.items():
        for intent in intents:
            reverse[intent] = dept
    return reverse


_INTENT_TO_DEPT = _build_reverse_map()


def get_department(intent: str) -> str:
    """Get the department for a given intent."""
    return _INTENT_TO_DEPT.get(intent, 'General Support')


def get_priority(intent: str, confidence: float = 1.0) -> str:
    """Assign ticket priority based on intent and prediction confidence.
    
    Args:
        intent: Predicted intent category.
        confidence: Model prediction confidence (0-1).
    
    Returns:
        Priority string: 'URGENT', 'HIGH', 'MEDIUM', or 'NORMAL'.
    """
    if intent in HIGH_PRIORITY_INTENTS and confidence >= 0.7:
        return 'URGENT'
    elif intent in HIGH_PRIORITY_INTENTS:
        return 'HIGH'
    elif intent in MEDIUM_PRIORITY_INTENTS:
        return 'MEDIUM'
    elif confidence < 0.3:
        return 'MEDIUM'  # Low confidence -> needs human review
    return 'NORMAL'


class TicketRouter:
    """Routes customer support tickets to appropriate departments."""
    
    def __init__(self, model, vectorizer, preprocessor_fn=None):
        """Initialize the router.
        
        Args:
            model: Fitted classifier.
            vectorizer: Fitted TF-IDF vectorizer.
            preprocessor_fn: Text preprocessing function (defaults to preprocess_text).
        """
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor_fn = preprocessor_fn or preprocess_text
    
    def route(self, text: str) -> dict:
        """Classify a ticket and route to the appropriate department.
        
        Args:
            text: Raw customer support text.
        
        Returns:
            dict with text, predicted_intent, department, confidence, priority, top_predictions.
        """
        cleaned = self.preprocessor_fn(text)
        features = self.vectorizer.transform([cleaned])
        
        # Get prediction
        intent = self.model.predict(features)[0]
        
        # Get confidence
        confidence = self._get_confidence(features)
        
        # Get top-5 predictions
        top_predictions = self._get_top_predictions(features, k=5)
        
        # Route
        department = get_department(intent)
        priority = get_priority(intent, confidence)
        
        return {
            'input_text': text,
            'preprocessed_text': cleaned,
            'predicted_intent': intent,
            'department': department,
            'confidence': round(confidence, 4),
            'priority': priority,
            'top_predictions': top_predictions,
        }
    
    def route_batch(self, texts: list) -> list:
        """Route multiple tickets at once."""
        return [self.route(text) for text in texts]
    
    def _get_confidence(self, features) -> float:
        """Get prediction confidence score."""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)
            return float(proba.max())
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features)
            # Normalize to 0-1 range using softmax-like approach
            exp_scores = np.exp(scores - scores.max())
            proba = exp_scores / exp_scores.sum()
            return float(proba.max())
        return 0.5  # Default
    
    def _get_top_predictions(self, features, k: int = 5) -> list:
        """Get top-k predictions with confidence scores."""
        classes = self.model.classes_
        
        if hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(features)[0]
        elif hasattr(self.model, 'decision_function'):
            raw = self.model.decision_function(features)[0]
            exp_scores = np.exp(raw - raw.max())
            scores = exp_scores / exp_scores.sum()
        else:
            return []
        
        top_indices = np.argsort(scores)[-k:][::-1]
        return [
            {'intent': classes[i], 'confidence': round(float(scores[i]), 4)}
            for i in top_indices
        ]


def get_routing_stats(y_true, y_pred) -> dict:
    """Compute routing statistics: how many tickets go to each department."""
    dept_counts = {}
    dept_correct = {}
    
    for true, pred in zip(y_true, y_pred):
        dept = get_department(pred)
        true_dept = get_department(true)
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        if dept == true_dept:
            dept_correct[dept] = dept_correct.get(dept, 0) + 1
    
    stats = {}
    for dept in dept_counts:
        stats[dept] = {
            'total_routed': dept_counts[dept],
            'correctly_routed': dept_correct.get(dept, 0),
            'routing_accuracy': round(dept_correct.get(dept, 0) / dept_counts[dept], 4)
                if dept_counts[dept] > 0 else 0.0,
        }
    
    return stats


def validate_routing_map() -> bool:
    """Validate that every intent maps to exactly one department."""
    all_intents = []
    for intents in ROUTING_MAP.values():
        all_intents.extend(intents)
    
    # Check for duplicates
    if len(all_intents) != len(set(all_intents)):
        dupes = [x for x in all_intents if all_intents.count(x) > 1]
        print(f"WARNING: Duplicate intents in routing map: {set(dupes)}")
        return False
    
    return True
