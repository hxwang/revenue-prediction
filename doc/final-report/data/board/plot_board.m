% #!/usr/bin/env octave -qf

f = figure;

% set(f, 'visible', 'off');
threash_hold = 2.6e6;

private_s = load('private_board.txt');
private_s = sort(private_s);
private_s = private_s(private_s<threash_hold);

public_s = load('public_board.txt');
public_s = sort(public_s);
public_s = public_s(public_s<threash_hold);

our_private_score = 1783082.67907;
our_private_rank = sum(private_s < our_private_score);

our_public_score = 1648696.38015;
our_public_rank = sum(public_s < our_public_score);

our_scores = [our_private_score, our_public_score];
our_ranks = [our_private_rank, our_public_rank];


plot(public_s, 'LineWidth', 2);
hold on;
plot(private_s, 'Color',[1.0,0.5,0.0], 'LineWidth', 4);
hold on;

plot(our_ranks, our_scores, 'rd', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

h = legend('Public Board','Private Board', 'Our Scores');

set(h, 'FontSize', 20, 'FontName', 'Helvetica', 'Location', 'Southeast');

grid on;

% title('Leaderboard', 'FontSize', 20, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Rank', 'FontSize', 20, 'FontName', 'Helvetica');
ylabel('RMSE', 'FontSize', 20, 'FontName', 'Helvetica');

ylim([1.4e6 threash_hold])

set(gca, 'FontSize', 20, 'FontName', 'Helvetica');

filename = '../../figs/pb.eps';

set(gcf,'paperposition',[0,0,8*1.5,4.8*1.5])

print('-depsc', filename);