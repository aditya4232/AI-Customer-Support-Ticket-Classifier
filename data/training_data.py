"""
Training data for the AI Customer Support Ticket Classifier.
Contains 100+ labeled examples across various categories.
"""

# Training data format: (ticket_text, category, priority, keywords)
TRAINING_DATA = [
    # Sales / Plan Upgrade (20 examples)
    ("I would like to upgrade my current plan to the premium package. Please let me know the steps and the price difference.", "Sales / Plan Upgrade", "Low", ["upgrade", "premium", "plan", "price"]),
    ("How can I upgrade my subscription to the enterprise tier?", "Sales / Plan Upgrade", "Low", ["upgrade", "subscription", "enterprise"]),
    ("What are the pricing options for your business plan?", "Sales / Plan Upgrade", "Low", ["pricing", "business", "plan"]),
    ("I want to switch from monthly to annual billing. What's the discount?", "Sales / Plan Upgrade", "Low", ["switch", "billing", "annual", "discount"]),
    ("Can you tell me what's included in the premium package?", "Sales / Plan Upgrade", "Low", ["premium", "included", "package"]),
    ("I need information about upgrading my account", "Sales / Plan Upgrade", "Low", ["upgrading", "account"]),
    ("What is the cost difference between basic and pro plan?", "Sales / Plan Upgrade", "Low", ["cost", "difference", "basic", "pro"]),
    ("Do you offer any discounts for non-profit organizations?", "Sales / Plan Upgrade", "Low", ["discounts", "non-profit"]),
    ("I'd like to know about your team pricing options", "Sales / Plan Upgrade", "Low", ["team", "pricing"]),
    ("What features come with the business plan?", "Sales / Plan Upgrade", "Low", ["features", "business", "plan"]),
    ("Can I get a trial of the premium features?", "Sales / Plan Upgrade", "Low", ["trial", "premium", "features"]),
    ("How do I change my subscription plan?", "Sales / Plan Upgrade", "Low", ["change", "subscription", "plan"]),
    ("What's the best plan for a small team of 5 people?", "Sales / Plan Upgrade", "Low", ["best", "small team"]),
    ("I want to upgrade but need invoice details", "Sales / Plan Upgrade", "Low", ["upgrade", "invoice"]),
    ("Are there any student discounts available?", "Sales / Plan Upgrade", "Low", ["student", "discounts"]),
    ("Can you explain the differences between your plans?", "Sales / Plan Upgrade", "Low", ["differences", "plans"]),
    ("I need to purchase more user licenses", "Sales / Plan Upgrade", "Medium", ["purchase", "licenses"]),
    ("What's the process to become a partner?", "Sales / Plan Upgrade", "Low", ["partner", "process"]),
    ("Do you offer white-label solutions?", "Sales / Plan Upgrade", "Low", ["white-label", "solutions"]),
    ("I want to bulk purchase credits", "Sales / Plan Upgrade", "Medium", ["bulk", "credits"]),
    
    # Technical Support (25 examples)
    ("I'm getting an error when trying to login. It says 'invalid credentials' but I'm sure my password is correct.", "Technical Support", "High", ["error", "login", "invalid credentials"]),
    ("The application keeps crashing when I try to export data", "Technical Support", "High", ["crashing", "export", "data"]),
    ("How do I reset my password?", "Technical Support", "Medium", ["reset", "password"]),
    ("I can't access my dashboard after the latest update", "Technical Support", "High", ["can't access", "dashboard", "update"]),
    ("The search function is not working properly", "Technical Support", "Medium", ["search", "not working"]),
    ("My files won't upload to the server", "Technical Support", "High", ["files", "won't upload"]),
    ("There's a bug in the reporting module", "Technical Support", "High", ["bug", "reporting"]),
    ("How can I integrate with your API?", "Technical Support", "Medium", ["integrate", "API"]),
    ("The mobile app keeps freezing on startup", "Technical Support", "High", ["mobile app", "freezing"]),
    ("I need help setting up two-factor authentication", "Technical Support", "Medium", ["two-factor", "authentication"]),
    ("The export to PDF is generating blank pages", "Technical Support", "Medium", ["export", "PDF", "blank"]),
    ("I'm experiencing slow performance with large datasets", "Technical Support", "Medium", ["slow", "performance", "datasets"]),
    ("How do I configure the webhook notifications?", "Technical Support", "Medium", ["configure", "webhook"]),
    ("The sync between devices is not working", "Technical Support", "High", ["sync", "not working"]),
    ("I need to migrate data from another platform", "Technical Support", "Medium", ["migrate", "data"]),
    ("The chatbot is giving incorrect responses", "Technical Support", "Medium", ["chatbot", "incorrect"]),
    ("Can you help me set up SSO?", "Technical Support", "Medium", ["SSO", "setup"]),
    ("The notifications are not being delivered", "Technical Support", "High", ["notifications", "not delivered"]),
    ("How do I create a backup of my data?", "Technical Support", "Low", ["backup", "data"]),
    ("There's a memory leak in the application", "Technical Support", "High", ["memory leak"]),
    ("I can't find the API documentation", "Technical Support", "Low", ["API documentation"]),
    ("The charts are not displaying correctly", "Technical Support", "Medium", ["charts", "displaying"]),
    ("How do I customize the email templates?", "Technical Support", "Low", ["customize", "email templates"]),
    ("The system is returning a 500 error", "Technical Support", "High", ["500 error"]),
    ("I need help with SSL certificate installation", "Technical Support", "High", ["SSL", "certificate"]),
    
    # Billing / Payments (20 examples)
    ("I was charged twice for my subscription this month", "Billing / Payments", "High", ["charged", "twice", "subscription"]),
    ("How do I update my credit card information?", "Billing / Payments", "Medium", ["update", "credit card"]),
    ("I need a refund for my last purchase", "Billing / Payments", "High", ["refund", "purchase"]),
    ("Why was my bill higher than expected?", "Billing / Payments", "Medium", ["bill", "higher"]),
    ("Can I get an invoice for tax purposes?", "Billing / Payments", "Low", ["invoice", "tax"]),
    ("I want to change my payment method to PayPal", "Billing / Payments", "Medium", ["payment method", "PayPal"]),
    ("My promo code isn't working at checkout", "Billing / Payments", "Medium", ["promo code", "not working"]),
    ("I need to cancel my subscription", "Billing / Payments", "Medium", ["cancel", "subscription"]),
    ("Where can I find my billing history?", "Billing / Payments", "Low", ["billing history"]),
    ("The automatic payment failed, what do I do?", "Billing / Payments", "High", ["automatic payment", "failed"]),
    ("I need to change my billing address", "Billing / Payments", "Low", ["billing address"]),
    ("Can I get a receipt for my payment?", "Billing / Payments", "Low", ["receipt", "payment"]),
    ("Why am I being charged a processing fee?", "Billing / Payments", "Medium", ["processing fee"]),
    ("I want to switch to annual billing for a discount", "Billing / Payments", "Low", ["annual billing", "discount"]),
    ("My account shows overdue but I paid", "Billing / Payments", "High", ["overdue", "paid"]),
    ("How do I enable automatic renewal?", "Billing / Payments", "Low", ["automatic renewal"]),
    ("I need to request a pro-rated charge", "Billing / Payments", "Medium", ["pro-rated"]),
    ("The tax calculation seems incorrect", "Billing / Payments", "Medium", ["tax", "incorrect"]),
    ("Can I get a detailed breakdown of charges?", "Billing / Payments", "Low", ["breakdown", "charges"]),
    ("I want to upgrade but keep my current billing cycle", "Billing / Payments", "Low", ["upgrade", "billing cycle"]),
    
    # Account (20 examples)
    ("I can't login to my account - it says account locked", "Account", "High", ["can't login", "account locked"]),
    ("How do I change my email address?", "Account", "Medium", ["change", "email"]),
    ("I need to update my profile information", "Account", "Low", ["update", "profile"]),
    ("My account was hacked, what should I do?", "Account", "High", ["hacked", "account"]),
    ("How do I delete my account?", "Account", "Medium", ["delete", "account"]),
    ("I need to add another user to my team", "Account", "Medium", ["add user", "team"]),
    ("Can I have two accounts with the same email?", "Account", "Low", ["two accounts"]),
    ("I forgot my password and can't reset it", "Account", "High", ["forgot password", "can't reset"]),
    ("How do I change my username?", "Account", "Low", ["change username"]),
    ("I want to transfer ownership of my account", "Account", "Medium", ["transfer", "ownership"]),
    ("My profile picture isn't displaying", "Account", "Low", ["profile picture"]),
    ("I need to set up account permissions for my team", "Account", "Medium", ["permissions", "team"]),
    ("Can I merge two accounts together?", "Account", "Medium", ["merge", "accounts"]),
    ("I need to reactivate my old account", "Account", "Medium", ["reactivate", "account"]),
    ("How do I enable dark mode in settings?", "Account", "Low", ["dark mode"]),
    ("I want to change my notification preferences", "Account", "Low", ["notification preferences"]),
    ("Can I use single sign-on with Google?", "Account", "Low", ["single sign-on", "Google"]),
    ("I need to verify my email address", "Account", "Medium", ["verify", "email"]),
    ("My account shows as inactive", "Account", "High", ["inactive", "account"]),
    ("How do I download my data?", "Account", "Low", ["download", "data"]),
    
    # General Inquiry (15 examples)
    ("What are your business hours?", "General Inquiry", "Low", ["business hours"]),
    ("Do you offer training sessions for new users?", "General Inquiry", "Low", ["training", "new users"]),
    ("Where can I find the user manual?", "General Inquiry", "Low", ["user manual"]),
    ("What languages do you support?", "General Inquiry", "Low", ["languages"]),
    ("Do you have a mobile app?", "General Inquiry", "Low", ["mobile app"]),
    ("What is your response time for support tickets?", "General Inquiry", "Low", ["response time"]),
    ("Are you GDPR compliant?", "General Inquiry", "Low", ["GDPR", "compliant"]),
    ("Do you offer on-premise installation?", "General Inquiry", "Medium", ["on-premise"]),
    ("What integrations do you support?", "General Inquiry", "Low", ["integrations"]),
    ("Where can I leave feedback?", "General Inquiry", "Low", ["feedback"]),
    ("Do you have a community forum?", "General Inquiry", "Low", ["community forum"]),
    ("What is your uptime guarantee?", "General Inquiry", "Low", ["uptime"]),
    ("Are there any upcoming features?", "General Inquiry", "Low", ["upcoming features"]),
    ("Do you offer discounts for educational institutions?", "General Inquiry", "Low", ["educational", "discounts"]),
    ("Where can I find your terms of service?", "General Inquiry", "Low", ["terms of service"]),
]

def get_training_data():
    """Return the complete training dataset."""
    return TRAINING_DATA

def get_categories():
    """Return all unique categories."""
    return list(set([item[1] for item in TRAINING_DATA]))

def get_priorities():
    """Return all unique priorities."""
    return list(set([item[2] for item in TRAINING_DATA]))

if __name__ == "__main__":
    print(f"Total training examples: {len(TRAINING_DATA)}")
    print(f"\nCategories: {get_categories()}")
    print(f"\nPriorities: {get_priorities()}")
