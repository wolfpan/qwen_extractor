#!/usr/bin/perl
use strict; use warnings; use JSON::PP;
binmode(STDOUT, ':utf8');
my $file = shift or die "usage: dump_nb.pl <ipynb> [maxchars]\n";
my $max = shift // 400;
open my $fh, '<:raw', $file or die $!;
local $/; my $data = <$fh>; close $fh;
my $nb = JSON::PP->new->utf8->decode($data);
my $i = 0;
for my $c (@{$nb->{cells}}) {
  my $src = join '', @{$c->{source} // []};
  my $type = $c->{cell_type};
  print "=" x 70, "\n[CELL $i] $type\n", "-" x 70, "\n";
  if (length $src > $max && $type ne 'markdown') { print substr($src,0,$max), " ...[TRUNC ", length($src), " chars]\n"; }
  else { print $src, "\n"; }
  for my $o (@{$c->{outputs} // []}) {
    my $ot = $o->{output_type} // '?';
    my $txt = '';
    if (exists $o->{text}) { $txt = join '', @{$o->{text}}; }
    elsif (exists $o->{data}{'text/plain'}) { $txt = join '', @{$o->{data}{'text/plain'}}; }
    elsif ($ot eq 'error') { $txt = ($o->{ename}//'') . ': ' . ($o->{evalue}//''); }
    my $nimg = exists $o->{data}{'image/png'} ? ' [HAS PNG IMAGE]' : '';
    print "--- output ($ot)$nimg ---\n";
    print length($txt) > $max*2 ? substr($txt,0,$max*2) . " ...[TRUNC " . length($txt) . " chars]\n" : "$txt\n" if length $txt;
  }
  $i++;
}
