#!/usr/bin/perl -w

# USAGE:  perl collect_pdf.pl 

use strict;
use diagnostics;

# settings:
my $resultf = "all_figures.tex";
my $latexcommand = "pdflatex";
my $folder_template = $ARGV[0] || "*" ;

# the length of figures:
my %figuresize = 
(
    "q2d"  => 0.9,
 "utie" => 0.5,
 "vtie" => 0.5,
 "small"=>0.3,
);

my $max_lines_on_page = 4;


# insert tex-file beginning:
system ("cat begin.tex > $resultf");
open(TEX, ">> $resultf");

my $string;
my @files;

#subroutine to get short filename:
sub shrink_name {
    my $wholename=$_[0];
    my @name = split(/\//, $wholename);
    return $name[-1];
}

my $line_count = 1; # counter to split subfigures between pages: only $max_lines_on_page lines of figures fit  in one page
my $row_fig_count = 0;

sub check_plot_count {
    # split figures:
    my $count = $_[0];
    if ($count % $max_lines_on_page == 0) {
        print TEX "\\end{figure}\n\\begin{figure}\n\\ContinuedFloat\n";
        print "New page\n";
    }
    return ++$count;
    
}

# subroutine to plot figures from one folder:
sub collect_figures {
    my $dir = $_[0];
    print ("...collecting figures from $dir\n");
    ############################
    # square  figures
    
#     print "Begin of fig $dir \n";<STDIN>;
    
    print TEX "\\begin{figure}[h!]\n\\centering\n";
    @files = `ls $dir/*png`;
    
    my $fs_previous;
    foreach my $png_file (@files) {
        chomp($png_file);
#                      print $png_file; <STDIN>;
        my $fs;
        my $name = shrink_name ($png_file);
        
        my $found_match=0;
        foreach my $type (keys %figuresize) {
            #                 print $type; <STDIN>;
            
            if ($png_file =~ /$type/) {
                #                     print "$png_file equals: ", $type; <STDIN>;
                $fs = $figuresize{$type};
                $found_match++;
            }
        }
        if ($found_match==0) {
            #                 print "$png_file equals small"; <STDIN>;
            $fs = $figuresize{small};
        } else {
            $found_match=0;
        }
        printf TEX "\\begin{subfigure}[b]{$fs\\textwidth}\n\\includegraphics[scale=0.5]{$png_file}\n\\cprotect\\caption{\\verb:$name:}\n\\end{subfigure}\n\\hfill\n";
        $row_fig_count++;
        #             print "row_fig_count=",$row_fig_count; <STDIN>;
        
        #             print "math: " ,1 / $fs,  " integer: ", int(1 / $fs); <STDIN>;
        #             print $row_fig_count % (int(1 / $fs)); <STDIN>;
        
        if ($row_fig_count % (int(1 / $fs)) ==0 ) {
            $line_count = check_plot_count($line_count);
        } 
        if ($fs_previous != $fs) {$line_count++};
#          print "line_count=",$line_count; <STDIN>;
         $fs_previous = $fs;
    }
    
    
    printf(TEX "\\cprotect\\caption{{\\verb:$dir:}}\n\\label{fig:}\n\\end{figure}\n");
#     print "End of fig\n";<STDIN>;
    
}




# get names of folders with figures:
my $list_default = `ls -d $folder_template*/`; 

print "Collect figures from all folders:\n";
print $list_default;

my @collection =  split(/\n/, $list_default); 
foreach my $dir (@collection) {
    collect_figures ($dir);
}

# insert end of the tex-file:
close(TEX);
system ("cat end.tex >> $resultf");

# compile Latex:
system ("$latexcommand $resultf");
exit; 

