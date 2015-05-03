#!/usr/bin/env octave

s = load('scores.txt');
s = sort(s);
s = s(s<3e6);
f = figure
set(f, "visible", "off")
plot(s)
axis tight
grid on
print("score.png", "-dpng")