import WallGo


temp = WallGo.WallGoResults()
print(temp.wallVelocity)


def fn_update(res):
    res.wallVelocity = 3.7


fn_update(temp)
print(temp.wallVelocity)