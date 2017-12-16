using Plots
function test_recon()
x = 0:.1:pi
xh = x +.05
y = sin(x)
r = copy(y)
l = copy(y)

recon!(l,r,y)

plot(sin, 0,pi, label="sin", title="points should be on curve")
scatter!(x, y, label="y")
scatter!(xh, l, label="l")
scatter!(xh, r, label="r")
end
