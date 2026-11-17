\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{clear2027}[2024/07/20 v1.0 CLeaR 2027 Proceedings]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{323}  % will need to be updated for the final submissions
\jmlryear{2027}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\ifanonsubmission
 \newcommand{\clearauthor}[1]{}
 \author{Author names withheld}
 \editor{Under Review for CLeaR 2027}
 \jmlrworkshop{6th Conference on Causal Learning and Reasoning}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\clearauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Karthika Mohan and Shohei Shimizu}
  \jmlrworkshop{6th Conference on Causal Learning and Reasoning}
 \fi
\fi

\newenvironment{extendedabstract}
{{\centering\large\bfseries Extended Abstract\par}\vspace{0.7ex}%
\bgroup
   \leftskip 20pt\rightskip 20pt\small\noindent\ignorespaces}%
{\par\egroup\vskip 0.25ex}

\newenvironment{fullversion}
{\bgroup\leftskip 20pt\rightskip 20pt \small\noindent{\bfseries
Full version:} \ignorespaces}%
{\par\egroup\vskip 0.25ex}

