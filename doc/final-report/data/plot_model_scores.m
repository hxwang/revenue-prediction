#!/usr/bin/env octave -qf

figure('visible','off');

cv_scores = [2.31220; 2.30149; 2.30379; 2.32837; 2.33207; 2.35433];
pb_scores = [1.64870; 1.76177; 1.73142; 1.90432; 1.80864; 1.80236];
model_names = ['Ensamble'; 'KNN'; 'NuSVR'; 'SVR'; 'GB'; 'RF'];

% sort by public board score
[B, I] = sort(pb_scores);

cv_scores = cv_scores(I);
pb_scores = pb_scores(I);
model_names = model_names(I, :);


scores = [cv_scores, pb_scores];

bar(scores);

% title('RMSE / 10^6 ', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
% xlabel('Models', 'FontSize', 16, 'FontName', 'Helvetica');
ylabel('RMSE / 10^6', 'FontSize', 16, 'FontName', 'Helvetica');

ylim([1.6 2.5]);

set(gca, 'XTickLabel', model_names);

h = legend({'Cross Validation', 'Public Board'});

set(h, 'FontSize', 16, 'FontName', 'Helvetica');

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

print('-depsc', '../figs/models.eps', '-S800,300');