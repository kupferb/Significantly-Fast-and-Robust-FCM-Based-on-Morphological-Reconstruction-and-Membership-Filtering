clearvars
close all

if strcmp(getenv('computername'),'BENNYK')
    bsdsRoot = 'C:\Users\Benny\MATLAB\Projects\AF-graph\BSD';
    addpath C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels\others
    addpath C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels\evals

else
    bsdsRoot = 'D:\MATLAB\github\AF-graph\BSD';
    addpath D:\MATLAB\github\Segmentation-Using-Superpixels\others
    addpath D:\MATLAB\github\Segmentation-Using-Superpixels\evals

end
fid = fopen('Nsegs.txt','r');
Nimgs = 300; % number of images in BSDS300
[BSDS_INFO] = fscanf(fid,'%d %d \n',[2,Nimgs]);
fclose(fid);
run_type = "train";
if run_type == "test"
    Nimgs = 100;
    test_ims_map = "ims_map_test.txt";
    fid = fopen(test_ims_map);
    test_ims_map_data = cell2mat(textscan(fid,'%f %*s'));
    fclose(fid);
    %%
    BSDS_INFO = BSDS_INFO(:,ismember(BSDS_INFO(1,:),test_ims_map_data));

elseif run_type == "train"
    Nimgs = 200;
    train_ims_map = "ims_map_train.txt";
    fid = fopen(train_ims_map);
    test_ims_map_data = cell2mat(textscan(fid,'%f %*s'));
    fclose(fid);
    %%
    BSDS_INFO = BSDS_INFO(:,ismember(BSDS_INFO(1,:),test_ims_map_data));

else
    Nimgs = 300;
end

Nimgs_inds = 1:Nimgs;
Nimgs = length(Nimgs_inds);
PRI_all = zeros(Nimgs,1);
VoI_all = zeros(Nimgs,1);
GCE_all = zeros(Nimgs,1);
BDE_all = zeros(Nimgs,1);
se=3; % the parameter of structuing element used for morphological reconstruction
w_size=3; % the size of fitlering window

for k_idxI = 1:Nimgs%64:Nimgs
    idxI = Nimgs_inds(k_idxI);
        img_name = int2str(BSDS_INFO(1,idxI));
    img_loc = fullfile(bsdsRoot,'images','test',[img_name,'.jpg']);    
    if ~exist(img_loc,'file')
        img_loc = fullfile(bsdsRoot,'images','train',[img_name,'.jpg']);
    end
    
    I = imread(img_loc);
    Nseg = 4;
    [~,U1,~,~]=FRFCM_c(double(I),Nseg,se,w_size);
    NcutEigenvectors = U1';
    evec = bsxfun( @rdivide, NcutEigenvectors, sqrt(sum(NcutEigenvectors.*NcutEigenvectors,2)) + 1e-10 );
%     evec = NcutEigenvectors;
    % k-means
    labels = k_means(evec',Nseg);
    label_img = reshape(labels,size(I(:,:,1)));

%     [f_seg,label_img]=fcm_image_color(I,U1);
    gt_imgs = readSegs(bsdsRoot,'color',str2double(img_name));
    out_vals = eval_segmentation(label_img,gt_imgs);
    fprintf('%6s: %2d %9.6f, %9.6f, %9.6f, %9.6f\n', img_name, Nseg, out_vals.PRI, out_vals.VoI, out_vals.GCE, out_vals.BDE);
    
    PRI_all(k_idxI) = out_vals.PRI;
    VoI_all(k_idxI) = out_vals.VoI;
    GCE_all(k_idxI) = out_vals.GCE;
    BDE_all(k_idxI) = out_vals.BDE;
end
fprintf('Mean: %14.6f, %9.6f, %9.6f, %9.6f \n', mean(PRI_all), mean(VoI_all), mean(GCE_all), mean(BDE_all));
