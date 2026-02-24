# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in miniAI, please report it responsibly by:

### Private Reporting (Preferred)

1. **GitHub Security Advisories:** [Report a vulnerability](https://github.com/nelsonramosua/miniAI/security/advisories/new)
2. **Email:** nelsonramos@ua.pt.

**Please do NOT create a public issue for security vulnerabilities.**

### What to Include

- Description of the vulnerability.
- Steps to reproduce.
- Affected versions.
- Potential impact.
- Suggested fix (if any).
- Your contact information (for follow-up) (if not by email).

---

## Response Timeline

| Stage | Timeline |
|-------|----------|
| **Initial response** | Within 48 hours |
| **Status update** | Within 1 week |
| **Fix development** | Depends on severity |
| **Security advisory** | With fix release |

**These response timelines can be subject to change.**

---

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | Yes             |
| < 0.0.1   | No              |

We only provide security updates for the latest release. \
Please upgrade to the latest version.

---

## Security Best Practices

When using miniAI:

### For Users
- Use the latest release version.
- Keep dependencies updated (check Dependabot alerts).
- Run with least privilege (don't run as root).
- Validate input data before training.
- Don't expose trained model files publicly (may contain training data patterns).

### For Contributors
- Follow secure coding practices.
- Never commit secrets or API keys.
- Test input validation thoroughly.
- Use tools like Valgrind to check for memory leaks.
- Review CodeQL security alerts.

### For Deployment
- Use Docker containers for isolation.
- Limit network access if not needed.
- Monitor for unusual resource usage.
- Keep host system updated.

---

## Known Security Considerations

### Memory Management
- miniAI uses custom arena allocators for performance.
- Bounded by `ARENA_SIZE` to prevent excessive memory allocation.
- All allocations are checked for NULL returns.

### Input Processing
- PNG image processing uses `stb_image` (bounds-checked).
- Input validation on command-line arguments.
- Grid size limits prevent buffer overflows.

### Model Files
- Binary format with version checking.
- Dimension validation on load.
- No executable code in model files.

---

## Security Features

**Memory safety:**
- Arena allocator with bounds checking.
- No dynamic allocation after initialization (except on image processing pipeline).
- Valgrind should be clean (no leaks, no undefined behavior).

**Input validation:**
- Argument parsing with bounds checks.
- Grid size validation.
- File path sanitization.

**Dependencies:**
- Minimal external dependencies (only OpenMP).
- `stb_image.h` single-header library (widely audited).

**Build security:**
- Compilation with `-Wall -Wextra`.
- Optional AddressSanitizer builds.
- Valgrind memory checks in CI.

---

## Disclosure Policy

Once a vulnerability is fixed:

1. We'll publish a **GitHub Security Advisory**.
2. We'll release a **patch version** with the fix.
3. We'll credit the reporter (unless they prefer anonymity).
4. We'll update this SECURITY.md if practices change.

---

## Hall of Fame

Security researchers who have responsibly disclosed vulnerabilities:

_(None yet - be the first!)_

---

## Questions?

For general security questions (not vulnerabilities), feel free to:
- Open a [Discussion](https://github.com/nelsonramosua/miniAI/discussions).
- Reach out via [email](nelsonramos@ua.pt).

Thank you for helping keep miniAI secure!