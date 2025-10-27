clear all;close all;clc
%% generate random samples
%n = 30;       % number of points
d = 3 ;         % dimension of the space
tol = 1e-03;            % tolerance value used in "inhull.m" (larger value high precision, possible numerical error)
 
for i = 1:2
points = zeros(8, 3); 
i
for ii = 1:2
    for j = 1:2
        for k = 1:2
            lower_bound_x = (ii - 1) / 2;
            upper_bound_x = ii / 2;
            lower_bound_y = (j - 1) / 2;
            upper_bound_y = j / 2;
            lower_bound_z = (k - 1) / 2;
            upper_bound_z = k /2;
            
            random_point_x = lower_bound_x + (upper_bound_x - lower_bound_x) * rand();
            random_point_y = lower_bound_y + (upper_bound_y - lower_bound_y) * rand();
            random_point_z = lower_bound_z + (upper_bound_z - lower_bound_z) * rand();
            
            point_index = (ii - 1) * 4+ (j - 1) * 2+ k;
            
            points(point_index, :) = [random_point_x, random_point_y, random_point_z];
        end
    end
end
    pos0 = points;
    bnd0 = [ 0 0 0;0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1 ; 1 1 1; 1 1 0] ;%rand(m,d);       % generate boundary point-candidates
    K = convhull(bnd0);
    bnd_pnts = bnd0(K,:);   % take boundary points from vertices of convex polytope formed with the boundary point-candidates
    %in = inhull(pos0,bnd0,[],tol); 
    pos = pos0;
    [vornb,vorvx] = polybnd_voronoi(pos,bnd_pnts);
    path = fullfile('H:\voronoi_0252\', 'data', ['stru_' num2str(i)]);
    mkdir(path);
    K1={};V=[];
 
        for ii = 1:size(pos,1)
        [K,v] = convhulln(vorvx{ii}); 
        K1{ii} = K;
        V(1,ii)= v;
        end
 
    fileName = 'idx.txt';
    filePath = fullfile(path, fileName);
    fid = fopen(filePath,'w');
    fprintf(fid,"%.5f  ",i);
    fclose(fid);
    
    save(['H:\voronoi_0522\data\stru_' num2str(i) '\V.mat'], 'V');
    save(['H:\voronoi_0522\data\stru_' num2str(i) '\K.mat'], 'K1');
    save(['H:\voronoi_0522\data\stru_' num2str(i) '\points.mat'], 'pos');
    save(['H:\voronoi_0522\data\stru_' num2str(i) '\vornb.mat'], 'vornb');
    save(['H:\voronoi_0522\data\stru_' num2str(i) '\vorvx.mat'], 'vorvx');
end

for i = 1:size(vorvx,2)
    col(i,:)= rand(1,3);
end
figure('position',[0 0 600 600],'Color',[1 1 1]);
for i = 1:size(pos,1)
K = convhulln(vorvx{i});
trisurf(K,vorvx{i}(:,1),vorvx{i}(:,2),vorvx{i}(:,3),'FaceColor',col(i,:),'FaceAlpha',0.5,'EdgeAlpha',1)
hold on;
end
scatter3(pos(:,1),pos(:,2),pos(:,3),'Marker','o','MarkerFaceColor',[0 .75 .75], 'MarkerEdgeColor','k');
axis('equal')
axis([0 1 0 1 0 1]);
set(gca,'xtick',[0 1]);
set(gca,'ytick',[0 1]);
set(gca,'ztick',[0 1]);
set(gca,'FontSize',16);
xlabel('X');ylabel('Y');zlabel('Z');
