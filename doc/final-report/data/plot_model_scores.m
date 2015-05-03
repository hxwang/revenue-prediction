#!/usr/bin/env octave -qf

figure('visible','off');

cv_scores = [2.31220; 2.30149; 2.30379; 2.32837; 2.33207; 2.35433];
pb_scores = [1.64870; 1.76177; 1.73142; 1.90432; 1.80864; 1.80236];
scores = [cv_scores, pb_scores];
model_names = ['Ensamble', 'KNN'; 'NuSVR'; 'SVR'; 'GB'; 'RF'];
bar(scores);

% title('RMSE / 10^6 ', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Models', 'FontSize', 16, 'FontName', 'Helvetica');
ylabel('RMSE / 10^6', 'FontSize', 16, 'FontName', 'Helvetica');

ylim([1.5 2.4]);

set(gca, 'XTickLabel', model_names);

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

print('-depsc', '../figs/models.eps');