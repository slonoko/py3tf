#print('Hello World')
#name = input('your name?\n')
#print(f'so your name is {name}')

pi = 3.14
r = 10
area = pi * r**2

print(area)

b = 10
h = 2
perimeter = 2 * (b+h)

print(perimeter)

print('id perimeter 1.1',id(perimeter), type(perimeter))

b = 6
perimeter = 2 * (b+h)

print(perimeter)

print('id perimeter 1.2',id(perimeter), type(perimeter))

b = 3
perimeter = 2 * (b+h)

print(perimeter)

print('id perimeter 1.3',id(perimeter), type(perimeter))


per2 = perimeter

print('id perimeter 2.1',id(per2), type(per2))

per2 = 'get'

print('id perimeter 2.2',id(per2), type(per2))

names = []

print('names:', names)
print('names id type:',id(names), type(names))

names.append('elie')

print('names:', names)
print('names id type: %s %s' % (id(names), type(names)))

print('iluvya!' * 10)

import operator

sum = operator.add(2, 4)

print('sum', sum)
sum = operator.add(2.1, 4.1)
print('sum', sum)


