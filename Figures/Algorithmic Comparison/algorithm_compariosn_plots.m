%% Comparison with Top-X performances for scenario 6
%Top-X    1       2       3       4       5  
data = [28.75	46.3	51.56	52.24	52.24; % LT
        29.43	49.01	66.25	71.04	71.3;  % KNN
        41.51	66.2	80.94	89.58	93.8]; % NN

% legend_entries = {'LT', 'KNN', 'NN'};
% legend_entries = {'NN', 'KNN', 'LT'};
legend_entries = {'Neural Network', 'K-Nearest Neighbors', 'Lookup Table'};

data = flipud(data);

close all;
f = figure();
hold on;

markers = {'p', 'o', 's'};
for i = 1:3
%     plot(data(i,:), '-o', 'MarkerFaceColor', 'auto', 'Linewidth', 1.2)
    h = plot(data(i,:), '-o', 'Linewidth', 1.2, 'Marker', markers{i});
    set(h, 'MarkerFaceColor', get(h,'Color')); 
end

legend(legend_entries, 'Location', 'nw')
ylabel('Top-k Beam Prediction Accuracy [%]')
xlabel('Beams (k)')
xticks(1:5)
grid on;
box on;
ax = gca(); ax.GridAlpha = 0.3;
print(f, 'top-k_algorithm_comparison4.eps', '-depsc')
saveas(f, 'top-k_algorithm_comparison4.svg')

%% Bar plot for all scenarios
%        NN      KNN     LT  
data = [55.57	49.45	40.69;
        48.86	45.77	44.26;
        31.09	28.91	25.79;
        29.14	26.12	22.54;
        43.12	45.19	38.73;
        41.51	29.43	28.75;
        27.82	24.71	21.88;
        43.65	40.1	37.12;
        38.73	37.03	35.04];

close all
f = figure(1);
bar_handle = barh(data, 'FaceAlpha', 0.7, 'EdgeAlpha', 0.0);
% axis tight;
% legend_h = legend(legend_entries)
% legend(fliplr(bar_handle), {'LT', 'KNN', 'NN'});
legend(fliplr(bar_handle), {'Lookup Table', 'K-Nearest Neighbors', 'Neural Network'},...
       'location', 'NE');
% Remove tick marks without removing the labels
ax = gca();
set(ax,'ytick',[])
% ax = gca(); % ax.YAxis.Visible = 'off';
ytickvalues = 1:9;
x = zeros(size(ytickvalues)) - 0.5;
str = string(ytickvalues);
text(x, ytickvalues, str, 'HorizontalAlignment', 'right');

xlabel('Top-1 Beam Prediction Accuracy [%]')
ylabel({'Scenarios',''})
grid on; 
grid minor
ax.GridAlpha = 0.3;
print(f, 'bars3.eps', '-depsc')
saveas(f, 'bars3.svg')