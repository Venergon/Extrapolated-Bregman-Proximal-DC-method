function M = read_bmp(name)

// read_bmp - read a 8 bit bw binary file
//
//   M = read_bmp(name);
//
//   Copyright (c) 2008 Gabriel Peyre

if isempty((strfind(name, '.')))
    name = [name '.bmp'];
end

fid = mopen(name, 'rb');
if fid<0
    error(['File ' name ' does not exists.']);
end

// sizes
[n,p,toReachMOf4] = read_header(fid);

// read data
[M,cnt] = mtlb_fread(fid, Inf, 'us');
mclose(fid);

nchannels = 3;
L = n*p*nchannels;

if length(M)<L
    error('size error');
end

M = reshape( M(end-L+1:end), nchannels, n, p );
M = permute(M, [3 2 1]);
// M = M';
M = M(size(M,1):-1:1,:,:);

endfunction

////////////////////////////////////////////////////////////////////

function [width,height,toReachMOf4] = read_header(fid)

//first two bytes should be 'BM'
// bytes 0x 0-1
BM = mtlb_fread(fid,2,'us');
if(BM(1) ~= 66 | BM(2) ~= 77)
	warning(['Possible File Corruption/Invalid File format. ',...
				'File Indicator (''BM'') not present in bytes 0x 00-01.']);
end



//the file size is the number of bytes in the entire file. Not all renderers
//use it, but it's best to have, since many (like Window Previewer) do.
// b 0x 2-5
fileSize = mtlb_fread(fid,1,'ul');  % ul->ul
if (fileSize == 0)
	warning(['Poor Formatting Practice: Size of File should be ',...
		'specified in bytes 0x 02-05']);
end

//The next 4 bytes are reserved and should be set to 0. 
// b 0x 6-9
reserved = mtlb_fread(fid,1,'ul');
if(reserved ~= 0)
	warning(['Possible File Corruption/Invalid File Format. ',...
				'Reserved bytes 0x 06-09 should be set to zero.']);
end

//Next four bytes evaluate to the offset (in bytes from the beginning of the
//file = 0) of the image data
// b 0x 0A-0D
offset = mtlb_fread(fid,1,'ul');

//Size of Info Header. Lke the size of the file, some renderers don't care
//about this, but it should be there for the ones who do. It gives the size
//of the InfoHeaderStructure (not to be confused with the
//FileHeaderStructure through which we are currently stepping) in bytes.
// b 0x 0E-11
InfoHeaderSize = mtlb_fread(fid,1,'ul');
if (InfoHeaderSize == 0)
	warning(['Poor Formatting Practice: Size of InfoHeader should be ',...
		'specified in bytes 0x 0E-11']);
end

//the width of the bitmap image (in pixels).
// b 0x 12-15
width = mtlb_fread(fid,1,'ul');

//the height of the bitmap image (in pixels).
// b 0x 16-19
height = mtlb_fread(fid,1,'ul');

//the number of "planes" in the bitmap. I'm not entire clear on what that
//means, but it's almost always 1 for standard bitmaps
planes = mtlb_fread(fid,1,'ubit16');

//the number of bits used to represent each pixel.
// b 0x 1C-1D 
bitsPerPixel = mtlb_fread(fid,1,'ubit16');

//figure out how we'll have to read each pixel component (r,g, and b).
bitsPerComponent = bitsPerPixel/3;

//Number of bytes written to each row must be a mutliple of 4. 0's are
// appended to end to make it fit.
bytesPerRow = width*bitsPerPixel/8;
if mod(bytesPerRow,4) == 0
	toReachMOf4 = 0;
else
	toReachMOf4 = 4 - mod(bytesPerRow,4);
end

//the true width (in bytes) of each row that will be read.
twidth = width + toReachMOf4;

// the total number of bytes in the Image data.
bytesInImage = width*height*bitsPerPixel/8 + height*toReachMOf4;

//The compression. This function can currently only read uncompressed files.
// b 0x 1E-21
compression = mtlb_fread(fid,1,'ul');

//Size of the image data section in bytes. Like FileSize and InfoHeader
//Size, this isn't always needed, but should be there.
// b 0x 22-25
imageDataSize = mtlb_fread(fid,1,'ul');
if (imageDataSize == 0)
	// warning(['Poor Formatting Practice: Size of Image Data should be ',...
	//	'specified in bytes 0x 22-25']);
end

//Bits per meter wide. Don't ask me why this would be important. I don't
//know of any renderers who require this, or any encoders who write it.
// b 0x 26-29
bitsPerMeterWide = mtlb_fread(fid,1,'ul');

//Bits per meter high. Don't ask me why this would be important. I don't
//know of any renderers who require this, or any encoders who write it.
// b 0x 2A-2D
bitsPerMeterHigh = mtlb_fread(fid,1,'ul');

//Number of colors used. not real clear on this one yet. I think it's only
//needed for precision other than 24 bit, so it will be more fully
//implemented at another time.
// b 0x 2E-31
colorsUsed = mtlb_fread(fid,1,'ul');

//Number of important colors used. Again, not real sure about this, it has
//to do with specifying a color table for precision other than 24 bit. Look
//for it in future versions
// b 0x 32-35
importantColors = mtlb_fread(fid,1,'ul');

//we may or may not be at the beginning of the image data now. Until I
//figure out how to read color table and all that good stuff, we'll just
//skip over whatever's left.

//so far we've read 54 bytes (from 0-53 or 0x00 to 0x35);
//read the rest, up to the actual iamge data
filler = mtlb_fread(fid,offset-54,'us');

//colorUsed should specify either the size of the colortable, or be 0, in
//which case the length that we just read won't be less than it. 
if length(filler)<colorsUsed*4
    error(['There doesn''t appear to be a valid colortable here. If ',...
        'you are sure this is a valid bitmap (e.g., windows opens it) ',...
        'send it to me, so I can try to fix the bug']);
end

//at least some of the filler should be a color table, if specified. 
colorTable = filler(1:colorsUsed*4);
//fourth byte in each colortable row is reserved.
colorTable = reshape(colorTable,4,colorsUsed)';

endfunction