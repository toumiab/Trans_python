\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{java}[2007/12/03]

\RequirePackage{ifpdf}

\RequirePackage{etex}
% chargement de scrbook avec les bonnes options
\ifpdf
\LoadClass[pdftex,10pt,a4paper,twoside,liststotoc,bibtotoc,idxtotoc,headinclude]{book}

\RequirePackage[a4paper,vdivide={*,22cm,4cm}]{geometry}
%\LoadClass[pdftex,DIV12,10pt,a4paper,BCOR12mm,chapterprefix,liststotoc,bibtotoc,idxtotoc]{scrbook}
%\renewcommand*\sectfont{\normalcolor\rmfamily\bfseries}
%\setkomafont{title}{\rmfamily}
\else
%\LoadClass[dvips,DIV12,10pt,a4paper,twoside,BCOR12mm,chapterprefix,liststotoc,bibtotoc,idxtotoc,headinclude]{scrbook}
\LoadClass[dvips,10pt,a4paper,twoside,liststotoc,bibtotoc,idxtotoc,headinclude]{book}
\RequirePackage[a4paper,vdivide={*,22cm,4cm}]{geometry}
\fi




\RequirePackage{xcolor}
% jolis chapiters
\ifpdf
\RequirePackage{quot2}
\else
\definecolor{chaptergrey}{rgb}{0.4,0.4,0.4}
\RequirePackage[nogrey,utopia]{quotchap}
\fi
%\RequirePackage[Bjornstrup]{fncychap}


\RequirePackage{makeidx} % creation d'index
\RequirePackage{tabularx,longtable}
\RequirePackage{supertabular}
\RequirePackage{hhline}
\RequirePackage{eurosym} % pour le symbole euro
\RequirePackage{amsmath,amssymb,amsthm,euscript}
% graphicx avec la bonne option, le chemin des images 
% et les types d'images valides
\ifpdf
\RequirePackage[pdftex]{graphicx}
\graphicspath{{pdfs/}{pngs/}}
\DeclareGraphicsExtensions{.pdf,.png,.mps}
\else
\RequirePackage[dvips]{graphicx}
\graphicspath{{epss/}{texs/}{pss/}}
% choix des extensions pour le mode dvips
%\DeclareGraphicsExtensions{.eps}
\fi

\ifpdf
\pdfpageattr {/Group << /S /Transparency /I true /CS /DeviceRGB>>}
\fi


\RequirePackage{float}
% page de garde
\RequirePackage{pageGardeEnsta}


\RequirePackage{fnamed}
\RequirePackage{multicol}

\RequirePackage{tipa}
\RequirePackage{lettrine}

% modif des fontes pour le pdf
% \ifpdf
%   \RequirePackage{mathptmx}
%   \RequirePackage[scaled=.90]{helvet}
%   \RequirePackage{courier}
% \else
%   % fontes lmodern
%   \RequirePackage{lmodern}
% \fi
\RequirePackage{lmodern}
% aeguill correspond a ea plus les guillemets francais
% et on ajoute l'option cyr pour ecrire en cyrilique
\RequirePackage[cyr]{aeguill}
\RequirePackage{yfonts}
\newcommand{\enluminure}[2]{\lettrine[lines=3]{\small \initfamily #1}{#2}}
\RequirePackage[latin1]{inputenc} % pour taper les accents directement
\RequirePackage[annataritalic]{tengwarscript} % pour ecrire en elfique
\RequirePackage[frenchb]{babel} %  francais
%\RequirePackage[russian,frenchb]{babel} % style russe et francais
% pas d'espace automatiques avant les signes de ponctuation doubles !
\NoAutoSpaceBeforeFDP
% \frenchbsetup{AutoSpacePunctuation=false}  %frenchb recent
% paquetage ntheorem : permet de generer une table des theoremes
% attention : ntheorem a des bugs dans les algos de cesure (Overfull hbox)
%\RequirePackage[hyperref]{ntheorem}

\RequirePackage[french,undotted,tight]{minitoc}
%\RequirePackage[tight]{shorttoc}
\setcounter{tocdepth}{1}
%\setcounter{minitocdepth}{1}

%\RequirePackage{backref}

% hyperref doit etre charge en dernier
\ifpdf
\RequirePackage[pdftex,colorlinks]{hyperref}
\else
\RequirePackage[dvips]{hyperref}
\fi
%
\hypersetup{%
  pdftitle={Langage et algorithmique},
  pdfauthor={Rodéric Moitié, François Mekerke},
  pdfkeywords={java, algorithmique},
  bookmarksnumbered,
  pdfstartview={FitH},
  citecolor=blue,
}%

\unitlength 1cm
\RequirePackage{pgf,tikz}
\usetikzlibrary{automata,shapes,positioning,shadows,patterns,trees,arrows}


% ensembles
\RequirePackage{dsfont}
\def\nbR{\mathds{R}}
\def\nbN{\mathds{N}}
\def\nbQ{\mathds{Q}}
\def\Cle{\ensuremath{\EuScript K}}


\newtheorem{exemple}{Exemple}
\newtheorem{defi}{Définition}
\newtheorem{theo}{Théorème}
\newtheorem{lemme}{Lemme}
\newtheorem{corollaire}{Corollaire}
\newenvironment{preuve}[1] { {\bf Preuve
  } #1: \\} { \begin{flushright} $\blacksquare$ \end{flushright}}

\RequirePackage[toc,page]{appendix}


% \RequirePackage{algorithm}
% \floatname{algorithm}{Algorithme}
% \renewcommand{\listalgorithmname}{Liste des algorithmes}
% \RequirePackage{algpseudocode}
% \algblockdefx{Var}{EndVar}{\textbf{variables}}{\textbf{end variables}}
% \algblockdefx{Req}{EndReq}{\textbf{require :}}{\textbf{end require}}
% %\setlength{\algorithmicindent}{1cm}
% \algrenewcommand\algorithmicindent{1cm}


\RequirePackage[algo2e,lined,algoruled,french]{algorithm2e}

\SetKwIF{If}{ElseIf}{Else}{si}{alors}{sinon si}{sinon}{fin si}
\SetKwFor{For}{pour}{faire}{fin pour}
\SetKwFor{While}{tant que}{faire}{fin tant que}
\SetKwFor{ForAll}{pour tous}{faire}{fin pour tous}
\SetKwInput{KwData}{Données}
\SetKwInput{KwIn}{Entrées}

\RequirePackage{listings}
\definecolor{green_comment}{rgb}{0.25,0.5,0.5}
\lstset{  basicstyle=\small\ttfamily\upshape,
          keywordstyle=\color{blue}\bfseries,
          identifierstyle=,
          commentstyle=\ttfamily\color{green_comment},
          stringstyle=\ttfamily\color{red},
          showstringspaces=false,
          numbers=left,
          firstnumber=1,
          numberstyle=\tiny,
%          stepnumber=5,
          frame=single,
          postbreak=\space, breakindent=5pt, breaklines=true}



%%% cesures dans les references biblio avec apalike
% ne fonctionne qu'en pdf
\makeatletter
\let\@cite@ofmt\relax
\makeatother 

\makeatletter
% espace avant les ":" dans les figures
\renewcommand{\fnum@figure}{\figurename~\thefigure~}
% espace avant les ":" dans les listings
\renewcommand{\fnum@lstlisting}{\lstlistingname
  \ifx\lst@@caption\@empty\else~\thelstlisting~\fi}%
\makeatother

%%% Workaround for \lstlistoflistings by Markus Kohm
% \makeatletter% --> De-TeX-FAQ
% \renewcommand*{\lstlistoflistings}{%
%   \begingroup
%     \if@twocolumn
%       \@restonecoltrue\onecolumn
%     \else
%       \@restonecolfalse
%     \fi
%     \lol@heading
%     \setlength{\parskip}{\z@}%
%     \setlength{\parindent}{\z@}%
%     \setlength{\parfillskip}{\z@ \@plus 1fil}%
%     \@starttoc{lol}%
%     \if@restonecol\twocolumn\fi
%   \endgroup
% }
% \makeatother% --> \makeatletter 

\RequirePackage{nota}

%-----------------------------------------------------------------------
%
% configuration de l'environnement nota
%
\ifpdf
\newcommand{\ficnota}{danger}
\newcommand{\ficnotahack}{dosdane}
\newcommand{\ficnote}{information}
\else
\newcommand{\ficnota}{attention}
\newcommand{\ficnote}{note}
\newcommand{\ficnotahack}{question}
\fi

\newenvironment{nota}{\begin{pictonote}{\ficnota}}{\end{pictonote}}
\newenvironment{notahack}{\begin{pictonote}{\ficnotahack}}{\end{pictonote}}
\newenvironment{note}{\begin{pictonote}{\ficnote}}{\end{pictonote}}
%-----------------------------------------------------------------------

\RequirePackage{arabtex} % qq mots en arabe
\setcounter{secnumdepth}{2}
