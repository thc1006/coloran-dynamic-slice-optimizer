# Contributing to ColO-RAN Dynamic Slice Optimizer

We welcome contributions to the ColO-RAN Dynamic Slice Optimizer project! By contributing, you help us improve this GPU-accelerated optimization system for 5G network slicing.

## How to Contribute

### 1. Fork the Repository
Fork the `coloran-dynamic-slice-optimizer` repository on GitHub.

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/coloran-dynamic-slice-optimizer.git
cd coloran-dynamic-slice-optimizer
```

### 3. Create a New Branch
Create a new branch for your feature or bug fix. Use a descriptive name:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/fix-issue-description
```

### 4. Set Up Your Development Environment
Ensure you have Python 3.8+ installed. We recommend using a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements-dev.txt
pip install -e .
```

### 5. Make Your Changes
- Adhere to the existing code style (PEP 8, type hints, Google-style docstrings).
- Write clear, concise, and well-documented code.
- Ensure your changes are covered by tests.
- Run tests and linters before committing.

### 6. Run Tests
Before submitting your changes, ensure all tests pass:
```bash
pytest tests/
```
For GPU-specific tests, ensure you have a compatible GPU and CUDA setup:
```bash
pytest tests/gpu --gpu-required
```

### 7. Lint and Format Your Code
We use `flake8`, `black`, and `isort` for code quality and formatting:
```bash
flake8 src/
black src/ tests/
isort src/ tests/
```

### 8. Commit Your Changes
Write clear and concise commit messages. Follow the Conventional Commits specification (e.g., `feat: Add new feature`, `fix: Fix a bug`).
```bash
git add .
git commit -m "feat: Your descriptive commit message"
```

### 9. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 10. Create a Pull Request
- Go to your forked repository on GitHub and create a new pull request to the `main` branch of the original repository.
- Provide a detailed description of your changes.
- Reference any related issues.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [your-email@example.com].

## Security Vulnerabilities

If you discover a security vulnerability, please report it responsibly by emailing [your-security-email@example.com] instead of opening a public issue.
