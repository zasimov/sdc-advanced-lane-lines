clear all;

load metrics.mat;

p = figure();
hold on;

plot(curvature)
title('Radius of Curvature')
xlabel("frame #")
ylabel("radius of curvature")

print(p, "output_images/metrics/roc.png", '-dpng');

hold off;

p = figure();
hold on;

plot(offset)
title('Vehicle Offset')
xlabel("frame #")
ylabel("offset (meters)")

print(p, "output_images/metrics/offset.png", '-dpng');

hold off;

p = figure();
hold on;

plot(miss)
title('Count of misses in row')
xlabel("frame #")
ylabel("count of misses")

print(p, "output_images/metrics/misses.png", '-dpng');

hold off;
