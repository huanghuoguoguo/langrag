import re

class Cleaner:
    """Basic text cleaner."""
    
    def clean(self, text: str) -> str:
        """Clean text by removing excessive whitespace and null characters."""
        if not text:
            return ""
        # Remove null characters
        text = text.replace('\0', '')
        # Replace multiple spaces/newlines with single space (optional, depending on requirement)
        # For now, let's just strip and normalize newlines slightly
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
