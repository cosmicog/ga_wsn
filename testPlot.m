b = b.*-1;
figure('visible', 'off');
get(gcf, 'RendererMode');
get(gcf, 'Renderer');
axis equal;
grid on;
hold on;
x1 = 50;
y1 = 0;
plot(x1, y1, '.', 'markersize', 30, 'color', [.9 .1 .1]);
title( sprintf('base : x = %d \t -y = %d', x1, y1) );
xlabel('X Axis');
ylabel('-Y Axis');

for i = 1:250
    dist = 10000.0;
    for k = c
        if k ~= -1
            loc_dist = sqrt( (a(i)-a(k+1))^2 + (b(i)-b(k+1))^2 );
            if loc_dist < dist
                dist = loc_dist;
                x1 = a(k+1); y1 = b(k+1);
            end
        end
    end

    x2 = a(i); y2 = b(i);
    p1 = [x1 x2];
    p2 = [y1 y2];
    line(p1, p2, 'color', [0 0 0]);

end

for k = 1:250
    if k == c(1)+1 || k == c(2)+1 || k == c(3)+1 || k == c(4)+1 || k == c(5)+1 || k == c(6)+1 || k == -1 % ||  k == c(7)+1 || k == c(8)+1 || k == c(9)+1  || k == c(10)+1  || k == c(11)+1 || k == c(12)+1 || k == c(13)+1 || k == c(14)+1
        plot(a(k), b(k), '.', 'markersize', 30, 'color', [.9 0 .9]);
        txt = sprintf('%d', k-1);
        text(a(k)-1, b(k), txt, 'fontsize', 8);
    else
        plot(a(k), b(k), '.', 'markersize', 25, 'color', [0 .8 .9]);
        txt = sprintf('%d', k-1);
        text(a(k)-1, b(k), txt, 'fontsize', 7);
    end
end
axis([-5 105 -105 5]);
%set(gcf,'units','normalized','outerposition',[0 0 1 1]); % daha büyük resim için...
myaa;
imwrite(getfield(getframe(gca),'cdata'),'myaa.png');
close all;
