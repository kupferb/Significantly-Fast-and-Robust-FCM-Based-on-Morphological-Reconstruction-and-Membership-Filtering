clearvars
close all
data_set = "bsd500";
run_type = "test";
if strcmp(getenv('computername'),'BENNYK')
    base_ssp_path = 'C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels';
    if data_set=="bsd300"
        bsdsRoot = 'C:\Users\Benny\MATLAB\Projects\AF-graph\BSD';
    end
    data_dir = "C:\Study\runs\bsd\test\4_funcs_FH_rgb_0.8_300_250_120";
else
    base_ssp_path = 'D:\MATLAB\github\Segmentation-Using-Superpixels';
    if data_set=="bsd300"
        bsdsRoot = 'D:\MATLAB\github\AF-graph\BSD';
    elseif data_set == "bsd500"
        bsdsRoot = 'D:\DataSet\BSD\500\BSDS500\data';
        gt_seg_root = 'D:\DataSet\BSD\500\BSDS500\data\groundTruth';
    end
    data_dir = "D:\Study\runs\bsd500\results\test\4_funcs_FH_rgb_0.8_300_250";

end
addpath(fullfile(base_ssp_path,'others'))
addpath(fullfile(base_ssp_path,'evals'))

fid = fopen(sprintf('Nsegs_%s.txt',data_set),'r');
[BSDS_INFO] = fscanf(fid,'%d %d \n');
fclose(fid);
BSDS_INFO = reshape(BSDS_INFO,2,[]);
if data_set == "bsd300"
    if run_type == "test"
        Nimgs = 100;   
    elseif run_type == "train"
        Nimgs = 200;    
    end
elseif data_set == "bsd500"
    if run_type == "test"
        Nimgs = 200;   
    elseif run_type == "train"
        Nimgs = 300;    
    end

end

ims_map = sprintf("ims_map_%s_%s.txt",run_type,data_set);
fid = fopen(ims_map);
ims_map_data = cell2mat(textscan(fid,'%f %*s'));
fclose(fid);
fid = fopen(ims_map,'rt');
image_map = textscan(fid,'%s %s');
fclose(fid);
orig_ims_nums = image_map{1};
new_ims_nums = image_map{2};
BSDS_INFO = BSDS_INFO(:,ismember(BSDS_INFO(1,:),ims_map_data));

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
