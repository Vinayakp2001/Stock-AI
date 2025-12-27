# Contributing to Autonomous Trading Bot

Welcome to the Autonomous Trading Bot project! 🚀 We're building an intelligent trading system that aims to achieve 60-70% win rate through advanced machine learning and multi-factor analysis.

## 🎯 Project Vision

Transform our current stock prediction system into a fully autonomous trading bot that:
- Analyzes stocks using technical, fundamental, and sentiment data
- Makes automated trading decisions with high accuracy
- Learns and improves from every trade
- Manages risk to protect capital
- Achieves consistent 60-70% win rate

---

## 🚀 Quick Start for Contributors

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/stock-prediction-agent-sdk.git
cd stock-prediction-agent-sdk
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

### 3. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=trading --cov=backtesting

# Run specific test file
pytest tests/test_prediction_agent.py -v
```

### 4. Start the Dashboard
```bash
# Start the development dashboard
python app_fresh.py

# Access at http://localhost:8050
```

---

## 🏷️ How to Contribute

### 🟢 Good First Issues
Perfect for new contributors:
- Documentation improvements
- Adding unit tests
- Bug fixes in existing code
- Configuration enhancements
- UI/UX improvements

Look for issues labeled `🟢 good-first-issue`

### 🟡 Intermediate Issues
For contributors with some experience:
- New feature implementations
- Integration improvements
- Performance optimizations
- Data analysis enhancements

Look for issues labeled `🟡 intermediate`

### 🔴 Advanced Issues
For experienced contributors:
- Core trading engine components
- Machine learning model improvements
- Broker API integrations
- Risk management systems

Look for issues labeled `🔴 advanced`

### 🟣 Expert Issues
For domain experts:
- Trading strategy development
- Financial modeling
- Advanced ML techniques
- System architecture decisions

Look for issues labeled `🟣 expert`

---

## 📋 Development Process

### 1. Choose an Issue
1. Browse [open issues](https://github.com/YOUR_REPO/issues)
2. Read the issue description and acceptance criteria
3. Check if issue is assigned or has active discussion
4. Comment on the issue to express interest
5. Wait for maintainer approval before starting work

### 2. Create a Branch
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/issue-123-add-fundamental-analysis

# Branch naming convention:
# feature/issue-NUMBER-short-description
# bugfix/issue-NUMBER-short-description
# docs/issue-NUMBER-short-description
```

### 3. Development Guidelines

#### Code Style
- Follow PEP 8 for Python code
- Use type hints for all function parameters and returns
- Add docstrings to all classes and methods
- Keep functions small and focused (max 50 lines)
- Use meaningful variable and function names

#### Example Code Style:
```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Analyzes fundamental data for stock evaluation."""
    
    def __init__(self, cache_duration: int = 3600):
        """Initialize the fundamental analyzer.
        
        Args:
            cache_duration: Cache duration in seconds (default: 1 hour)
        """
        self.cache_duration = cache_duration
        self._cache: Dict[str, Any] = {}
    
    def calculate_pe_ratio(self, symbol: str) -> Optional[float]:
        """Calculate Price-to-Earnings ratio for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
            
        Returns:
            P/E ratio or None if data unavailable
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        try:
            # Implementation here
            pe_ratio = self._fetch_pe_ratio(symbol)
            logger.info(f"Calculated P/E ratio for {symbol}: {pe_ratio}")
            return pe_ratio
        except Exception as e:
            logger.error(f"Error calculating P/E for {symbol}: {e}")
            return None
```

#### Testing Requirements
- Write unit tests for all new functions
- Aim for 80%+ code coverage
- Include edge case testing
- Mock external API calls
- Add integration tests for complex features

#### Example Test:
```python
import pytest
from unittest.mock import Mock, patch
from agents.fundamental_agent import FundamentalAnalyzer

class TestFundamentalAnalyzer:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FundamentalAnalyzer()
    
    def test_calculate_pe_ratio_valid_symbol(self):
        """Test P/E calculation with valid symbol."""
        with patch.object(self.analyzer, '_fetch_pe_ratio', return_value=15.5):
            result = self.analyzer.calculate_pe_ratio('AAPL')
            assert result == 15.5
    
    def test_calculate_pe_ratio_invalid_symbol(self):
        """Test P/E calculation with invalid symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            self.analyzer.calculate_pe_ratio("")
    
    def test_calculate_pe_ratio_api_error(self):
        """Test P/E calculation when API fails."""
        with patch.object(self.analyzer, '_fetch_pe_ratio', side_effect=Exception("API Error")):
            result = self.analyzer.calculate_pe_ratio('AAPL')
            assert result is None
```

### 4. Commit Guidelines

#### Commit Message Format:
```
type(scope): short description

Longer description if needed

Fixes #123
```

#### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

#### Examples:
```bash
feat(fundamental): add P/E ratio calculation

Implement P/E ratio calculation using yfinance data.
Includes caching and error handling.

Fixes #1

fix(prediction): handle missing data in ML models

Add proper error handling when training data is insufficient.
Prevents crashes during model training.

Fixes #45

docs(readme): update installation instructions

Add detailed setup instructions for development environment.
Include troubleshooting section.

Fixes #67
```

### 5. Pull Request Process

#### Before Creating PR:
```bash
# Ensure all tests pass
pytest

# Check code style
flake8 agents/ trading/ backtesting/

# Check type hints
mypy agents/ trading/ backtesting/

# Update documentation if needed
# Add entry to CHANGELOG.md if significant change
```

#### PR Template:
```markdown
## 🎯 Description
Brief description of changes made.

## 📋 Changes Made
- [ ] Added fundamental analysis module
- [ ] Implemented P/E ratio calculation
- [ ] Added comprehensive tests
- [ ] Updated documentation

## 🧪 Testing
- [ ] All existing tests pass
- [ ] New tests added with 80%+ coverage
- [ ] Manual testing completed
- [ ] Integration tests pass

## 📚 Documentation
- [ ] Code comments added
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] CHANGELOG.md updated

## 🔗 Related Issues
Fixes #123
Related to #456

## 📸 Screenshots (if applicable)
[Add screenshots for UI changes]

## ✅ Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

#### PR Review Process:
1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Manual testing by reviewer if needed
4. **Approval**: PR approved by maintainer
5. **Merge**: Squash and merge to main branch

---

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                 # Unit tests
│   ├── test_agents/
│   ├── test_trading/
│   └── test_backtesting/
├── integration/          # Integration tests
│   ├── test_end_to_end/
│   └── test_api_integration/
└── fixtures/            # Test data and fixtures
    ├── sample_data/
    └── mock_responses/
```

### Test Categories

#### Unit Tests
- Test individual functions and methods
- Mock all external dependencies
- Fast execution (<1 second per test)
- High coverage (>90% for new code)

#### Integration Tests
- Test component interactions
- Use real data when possible
- Test error scenarios
- Moderate execution time (<30 seconds)

#### End-to-End Tests
- Test complete workflows
- Use paper trading environment
- Test with real market data
- Longer execution time (minutes)

### Running Tests
```bash
# Run all tests
pytest

# Run specific category
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=agents --cov-report=html

# Run specific test
pytest tests/unit/test_agents/test_fundamental_agent.py::TestFundamentalAnalyzer::test_calculate_pe_ratio

# Run tests in parallel
pytest -n auto
```

---

## 📚 Documentation Standards

### Code Documentation
- All classes must have docstrings
- All public methods must have docstrings
- Use Google-style docstrings
- Include type hints for all parameters
- Document exceptions that can be raised

### README Updates
- Update feature list when adding new capabilities
- Add configuration examples for new features
- Update installation instructions if dependencies change
- Add usage examples for new functionality

### API Documentation
- Use Sphinx for API documentation
- Generate docs automatically from docstrings
- Include code examples in documentation
- Keep documentation in sync with code

---

## 🚨 Issue Reporting

### Bug Reports
Use the bug report template and include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Screenshots if applicable

### Feature Requests
Use the feature request template and include:
- Clear description of the feature
- Business value and use cases
- Proposed implementation approach
- Acceptance criteria
- Dependencies and risks

### Performance Issues
Include:
- Performance benchmarks
- Profiling results if available
- System specifications
- Data size and complexity
- Comparison with expected performance

---

## 🎯 Development Priorities

### Phase 1: Foundation (Current)
**Priority**: 🔴 Critical
- Fundamental analysis module
- Sentiment analysis integration
- Multi-factor scoring system
- Risk management framework

### Phase 2: Trading System
**Priority**: 🟠 High
- Broker API integration
- Order execution engine
- Position management
- Paper trading system

### Phase 3: Advanced Features
**Priority**: 🟡 Medium
- Learning engine enhancements
- Portfolio optimization
- Real-time monitoring
- Alert systems

### Phase 4: Production
**Priority**: 🟢 Low
- Safety controls
- Comprehensive logging
- Performance optimization
- Mobile application

---

## 🏆 Recognition

### Contributor Levels

#### 🌟 First-time Contributor
- First merged PR
- Added to contributors list
- Welcome package and guidance

#### 🚀 Regular Contributor
- 5+ merged PRs
- Trusted with intermediate issues
- Invited to planning discussions

#### 🎯 Core Contributor
- 15+ merged PRs
- Can review other PRs
- Involved in architecture decisions

#### 👑 Maintainer
- 50+ merged PRs
- Deep project knowledge
- Can merge PRs and manage releases

### Recognition Methods
- Contributors listed in README
- Monthly contributor highlights
- Conference speaking opportunities
- Open source portfolio building
- LinkedIn recommendations

---

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Discord/Slack**: Real-time chat (link in README)
- **Email**: maintainers@tradingbot.com for private matters

### Mentorship Program
- New contributors paired with experienced mentors
- Weekly check-ins and guidance
- Code review and feedback
- Career development support

### Office Hours
- Weekly virtual office hours
- Direct access to maintainers
- Q&A sessions
- Architecture discussions

---

## 🔒 Security

### Reporting Security Issues
- **DO NOT** create public issues for security vulnerabilities
- Email security@tradingbot.com with details
- Include steps to reproduce
- Allow 90 days for fix before public disclosure

### Security Guidelines
- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow secure coding practices
- Regular dependency updates

---

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## 🙏 Thank You

Thank you for contributing to the Autonomous Trading Bot project! Your contributions help build a better, more intelligent trading system that can benefit the entire community.

Together, we're building the future of algorithmic trading! 🚀📈

---

**Happy Coding!** 💻✨