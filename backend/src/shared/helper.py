import re

class HelperFunction:
    @staticmethod
    def validate_date_format(value: str, optional: bool = True):
        """
        Validates and normalizes different date formats into yyyy-mm-dd.
        """
        if value is None or (not optional and value == "Not Specified"):
            raise ValueError("Value is missing")

        # if optional=True and "Not Specified", allow it
        if optional and value == "Not Specified":
            return value  

        # Date regex validation
        pattern1 = r'^(\d{2})-(\d{2})-(\d{4})$'   # dd-mm-yyyy
        pattern2 = r'^(\d{4})-(\d{2})-(\d{2})$'   # yyyy-mm-dd
        pattern3 = r'^(\d{2})\/(\d{2})\/(\d{4})$' # dd/mm/yyyy
        pattern4 = r'^(\d{4})\/(\d{2})\/(\d{2})$' # yyyy/mm/dd

        if re.search(pattern1, value):
            return re.sub(pattern1, r'\3-\2-\1', value)
        elif re.search(pattern2, value):
            return value
        elif re.search(pattern3, value):
            return re.sub(pattern3, r'\3-\2-\1', value)
        elif re.search(pattern4, value):
            return re.sub(pattern4, r'\1-\2-\3', value)
        else:
            raise ValueError("Invalid date format.")

    
    # =========================
    # Makes sure the value is a boolean
    # =========================
    @staticmethod
    def normalize_bool(value):
        if isinstance(value, bool):  # already a bool
            return value
        elif isinstance(value, str):
            val = value.strip().lower()
            if val in ["true", "1", "yes"]:
                return True
            elif val in ["false", "0", "no"]:
                return False
        raise ValueError("Invalid boolean value for paid field")
    
    # ========================
    # Validate password pattern
    # ========================
    # confirm symbol in password:
    @staticmethod
    def has_symbol(password:str):
        symbol_pattern = r"[!@#$%^&*]"
        if re.search(symbol_pattern,password):
            return
        else:
            raise ValueError("Password does not contain symbol.")
    # confirm digit in password:
    @staticmethod
    def has_digit(password:str):
        digit_pattern = r"[\d]"
        if re.search(digit_pattern,password):
            return
        else:
            raise ValueError("Password does not contain digit(s).")
    # confirm lower letter in password:
    @staticmethod
    def has_low_letter(password:str):
        lower_letter_pattern = r"[a-z]"
        if re.search(lower_letter_pattern,password):
            return
        else:
            raise ValueError("Password does not contain lowercase letters.")
    # confirm presence of upper case letters
    @staticmethod
    def has_upper_letter(password:str):
        upper_letter_pattern = r"[A-Z]"
        if re.search(upper_letter_pattern,password):
            return
        else:
            raise ValueError("Password does not contain uppercase letters.")
    # confirm general pattern:
    @staticmethod
    def pattern_match(password:str):
        if (len(password) > 8) or (len(password) < 4):
            raise ValueError("Password length out of range")
        pattern = r"^[a-zA-Z].{4,8}$"
        if not re.search(pattern,password):
            raise ValueError("Invalid password format")
        
    @staticmethod
    def validate_password_format(password:str):
        HelperFunction.has_symbol(password = password)
        HelperFunction.has_digit(password = password)
        HelperFunction.has_low_letter(password=password)
        HelperFunction.has_upper_letter(password=password)
        HelperFunction.pattern_match(password=password)