# Bank AI Chatbot

A sophisticated banking chatbot system with AI-powered conversational capabilities and secure authentication.

## Project Structure

```
bank-chatbot/
│
├── backend/
│   ├── Bank AI chatbot/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── chatbot.py
│   │   ├── inquiry_handler.py
│   │   ├── models.py
│   │   └── config.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/
│   │   │   ├── Auth/
│   │   │   └── Common/
│   │   ├── services/
│   │   ├── styles/
│   │   └── App.js
│   ├── package.json
│   └── README.md
│
└── README.md
```

## Backend Features

- **Secure Authentication**: JWT-based user authentication system
- **Real-time Chat**: WebSocket integration using Flask-SocketIO
- **AI Integration**: Advanced natural language processing for intent recognition
- **Banking Operations**: Handles account inquiries, loan status, and more
- **Session Management**: Robust chat session handling and state management
- **Data Persistence**: SQL database integration for messages and user data

## Frontend Features

- **Modern React UI**: Built with React.js and Material-UI components
- **Real-time Updates**: WebSocket integration for instant messaging
- **Responsive Design**: Mobile-first approach for all screen sizes
- **User Authentication**: Secure login/signup flows with JWT handling
- **Chat Interface**:
  - Message history with timestamp
  - Typing indicators
  - Message status (sent/delivered/read)
  - File attachment support
  - Quick reply buttons
- **Theme Customization**: Light/dark mode support
- **Error Handling**: Graceful error displays and retry mechanisms

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Initialize the database:
```bash
flask db upgrade
```

5. Run the backend server:
```bash
flask run
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API endpoints
```

3. Start development server:
```bash
npm start
```

## API Documentation

### Authentication Endpoints

- `POST /api/auth/register`: User registration
- `POST /api/auth/login`: User login
- `POST /api/auth/refresh`: Refresh access token
- `POST /api/auth/logout`: User logout

### Chat Endpoints

- `POST /api/chat/start`: Start new chat session
- `GET /api/chat/history`: Get chat history
- `WS /api/chat/ws`: WebSocket endpoint for real-time chat

## WebSocket Events

### Client Events
- `connect`: Initial connection
- `message`: Send chat message
- `typing`: User typing indicator
- `read`: Message read receipt

### Server Events
- `connected`: Connection confirmed
- `response`: Bot response
- `error`: Error message
- `typing`: Bot typing indicator

## Security Considerations

- JWT token authentication
- HTTPS encryption
- XSS prevention
- CSRF protection
- Rate limiting
- Input validation
- SQL injection prevention

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use ESLint for JavaScript
- Implement proper error handling
- Add comprehensive logging
- Write unit tests

### Git Workflow
- Feature branches
- Pull request reviews
- Semantic versioning
- Conventional commits

## Testing

### Backend Tests
```bash
python -m pytest
```

### Frontend Tests
```bash
npm test
```

## Deployment

### Backend Deployment
1. Set up production server (e.g., Ubuntu with Nginx)
2. Configure SSL certificates
3. Set up Gunicorn
4. Configure environment variables
5. Set up database backups

### Frontend Deployment
1. Build production bundle:
```bash
npm run build
```
2. Deploy to static hosting (e.g., Netlify, Vercel)
3. Configure environment variables
4. Set up CI/CD pipeline

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - See LICENSE file for details

## Support

For support, email: support@bankchatbot.com