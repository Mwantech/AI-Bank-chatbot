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
