def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.

    >>> to_smash(91)
    1
    """
    return total_candies % 3


# Check your answer

print (to_smash(4))