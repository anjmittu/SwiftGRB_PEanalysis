files=importdata('GRBlist.txt');
N=length(files);
sz=zeros(N,3);
wdw=zeros(N,2);
mm=zeros(N,2);
L=400;

% gkern=GaussNorm(-100:100);

datafile='GRBLCsmooth2_train.txt';
fptr=fopen(datafile,'w');
fprintf(fptr,'400,\n1,\n');

for i=1:N
    d=importdata([files{i} '.txt']);
    npts=length(d(:,1));
    j=1;
    while d(j,2)==0
        j=j+1;
    end
    sz(i,1)=j;
    j=npts;
    while d(j,2)==0
        j=j-1;
    end
    sz(i,2)=j;
    sz(i,3)=sz(i,2)-sz(i,1);
    pad1=ceil((L-sz(i,3))/2);
    pad2=L-pad1-sz(i,3)-1;
    wdw(i,:)=[sz(i,1)-pad1,sz(i,2)+pad2];
%     smoothed=conv(d(wdw(i,1):wdw(i,2),2),gkern,'same');
    smoothed=smoothn(d(wdw(i,1):wdw(i,2),2));
    smoothed=smoothed/max(smoothed);
    for j=1:L
        fprintf(fptr,'%f,',smoothed(j));
    end
    fprintf(fptr,'\n%d,\n',i);
    [mm(i,1),mm(i,2)]=minmax(d(:,2));
%     close all
%     figure(1)
%     plot(d(wdw(i,1):wdw(i,2),2),'-k')
%     hold on
%     plot(smoothed,'-b')
%     pause
end

fclose(fptr);
