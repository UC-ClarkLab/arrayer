function StackToVideo(stackname){

run("Image Sequence...", "open=[C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stackname\\20140311_stackname_w1_TimePoint_1.TIF] sort");

selectWindow("stackname");
run("Make Substack...", "delete slices=1-50");
saveAs("Tiff", "C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stacks\\stackname\\Substack_w1.tif");

selectWindow("stackname");
run("Make Substack...", "delete slices=1-50");
selectWindow("Substack (1-50)");
saveAs("Tiff", "C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stacks\\stackname\\Substack_w2.tif");

selectWindow("stackname");
run("Make Substack...", "delete slices=1-50");
saveAs("Tiff", "C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stacks\\stackname\\Substack_w3.tif");

selectWindow("stackname");
saveAs("Tiff", "C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stacks\\stackname\\Substack_w4.tif");


run("Merge Channels...", "c1=Substack_w3.tif c2=Substack_w2.tif c3=Substack_w1.tif c4=Substack_w4.tif create");
run("Stack to RGB", "slices");
saveAs("Tiff", "C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\stacks\\stackname\\CompositeStack_4channel.tif");

run("AVI... ", "compression=JPEG frame=7 save=[C:\\Users\\Brian\\Desktop\\Data For Analysis\\2014-06-02 BCP Z-stack + SKM\\20140311_Plate_3277\\video\\stackname_4channel_mov.avi]");

}
