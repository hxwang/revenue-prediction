#!/usr/bin/env octave -qf

s = dlmread('../train_scaled.csv');

printf('i\tcorr\n');
for i=1:41
    printf('%d\t%f\n', i, corr(s(:, i), s(:, 42)));
end