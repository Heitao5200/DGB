class Solution:
    def Fibonacci( n):

        if n<=0:
            return 0
        if n==1:
            return 1
        else:
            a,b=0,1
            for i in range(n-1):
                a,b=b,a+b
            return b

c=Solution.Fibonacci(9)
print(c)

class Solution:
    def jumpFloor( number):
        if number == 1:
            return 1
        if number == 2:
            return 2
        else:
            a,b =1,2
            # a = 1
            # b = 2
            for i in range(2, number ):
                a,b = b,a+b
            return b

d =Solution.jumpFloor(5)
print(d)