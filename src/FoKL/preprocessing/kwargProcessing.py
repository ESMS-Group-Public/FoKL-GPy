import warnings
__all__ = ['_process_kwargs', '_merge_dicts', '_set_attributes', '_str_to_bool']

def _str_to_bool(s):
    """Convert potential string (e.g., 'on'/'off') to boolean True/False. Intended to handle exceptions for keywords."""
    if isinstance(s, str):
        if s in ['yes', 'y', 'on', 'all', 'true', 'both']:
            s = True
        elif s in ['no', 'n', 'off', 'none', 'n/a', 'false']:
            s = False
        else:
            warnings.warn(f"Could not understand string '{s}' as a boolean.", category=UserWarning)
    elif s is None or not s:  # 'not s' for s == []
        s = False
    else:
        try:
            if s != 0:
                s = True
            else:
                s = False
        except:
            warnings.warn(f"Could not convert non-string to a boolean.", category=UserWarning)
    return s


def _process_kwargs(default, user):
    """Update default values with user-defined keyword arguments (kwargs), or simply check all kwargs are expected."""
    if isinstance(default, dict):
        expected = default.keys()
        if isinstance(user, dict):
            for kw in user.keys():
                if kw not in expected:
                    raise ValueError(f"Unexpected keyword argument: '{kw}'")
                else:
                    default[kw] = user[kw]
        else:
            raise ValueError("Input 'user' must be a dictionary formed by kwargs.")
        return default
    elif isinstance(default, list):  # then simply check for unexpected kwargs
        for kw in user.keys():
            if kw not in default:
                raise ValueError(f"Unexpected keyword argument: '{kw}'")
        return user
    else:
        raise ValueError("Input 'default' must be a dictionary or list.")


def _set_attributes(self, attrs):
    """Set items stored in Python dictionary 'attrs' as attributes of class."""
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            setattr(self, key, value)
    else:
        warnings.warn("Input must be a Python dictionary.")
    return


def _merge_dicts(d1, d2):
    """Merge two dictionaries into single dictionary in a backward-compatible way. Values of d2 replace any shared variables in d1."""
    d = d1.copy()
    d.update(d2)
    return d
