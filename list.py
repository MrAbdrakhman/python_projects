# r = ["Mario", "Bowser", "Luigi"]
# def purple_shell(racers):
#     """Given a list of racers, set the first place racer (at the front of the list) to last
#     place and vice versa.
#
#     >>> r = ["Mario", "Bowser", "Luigi"]
#     >>> purple_shell(r)
#     >>> r
#     ["Luigi", "Bowser", "Mario"]
#
#     """
#     #racers[-1], racers[0] = racers[0], racers[-1]
#     temp = racers[0]
#     racers[0] = racers[-1]
#     racers[-1] = temp
#     return racers
#
#
r=['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
#     """
#     d=len(arrivals)
#     if arrivals.index(name) < (d/2):
#         return 'fashionably NOT late'
#     return 'fashionably late'
#
# print (fashionably_late(r, 'Ford'))

multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
    print (product)
print(product)

s = 'steganograpHy is the practicE ofO conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')