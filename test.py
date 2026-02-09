
a=2800 #mm
b=670 #mm
v=0.3
E=210000
def defelction_rigity(h):
    D=(E*h**3)/(12*(1-v**2))
    return D
print(defelction_rigity(6.5))

def deflection(m,n):
    