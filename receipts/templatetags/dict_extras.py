from django import template

register = template.Library()

@register.filter
def dict_get(d, key):
    """
    Dictionary get filter for templates
    Usage: {{ my_dict|dict_get:my_key }}
    """
    if d is None:
        return ''
    return d.get(key, '')
