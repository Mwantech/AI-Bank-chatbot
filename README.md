# AI-Bank-chatbot



New-Item -ItemType File -Path "banking api/routes/__init__.py"
New-Item -ItemType File -Path "banking api/services/__init__.py"
New-Item -ItemType File -Path "banking api\__init__.py"

banking_api/
├── config.py
├── app.py
├── models/
│   └── models.py
├── services/
│   └── account_service.py
├── intent_patterns.json
├── nlp_service.py
└── response_cache.pkl
├── routes/
│   ├── account_routes.py
│   ├── customer_support_routes.py
│   ├── loan_routes.py
│   ├── location_routes.py
│   ├── investment_routes.py
│   └── chatbot_routes.py


your_project/
├── models/
│   ├── vectorizer.pkl
│   ├── patterns.pkl
│   └── intents.pkl
├── data/
│   ├── intent_patterns.json
│   └── response_cache.pkl
└── services/
    └── nlp_service.py




api endpoint for registering new user:http://localhost:5000/api/auth/register
{
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "password": "Password123!",
    "phoneNumber": "1234567890",
    "dateOfBirth": "1990-01-01",
    "address": "123 Main St, Springfield",
    "identificationNumber": "A12345678",
    "securityQuestion": "What is your favorite color?",
    "securityAnswer": "Blue"
}
