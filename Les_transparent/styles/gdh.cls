\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gdh}[2008/10/20]

\RequirePackage{ifpdf}

% chargement de scrbook avec les bonnes options
\ifpdf
\LoadClass[dvips,10pt,a4paper,twoside,liststotoc,bibtotoc,idxtotoc,headinclude]{book}
\RequirePackage[a4paper,vdivide={*,22cm,4cm}]{geometry}
%\LoadClass[dvips,DIV12,10pt,a4paper,twoside,BCOR12mm,chapterprefix,liststotoc,bibtotoc,idxtotoc,headinclude]{scrbook}
%\renewcommand*\sectfont{\normalcolor\rmfamily\bfseries}
%\setkomafont{title}{\rmfamily}
\else
%\LoadClass[pdftex,DIV12,10pt,a4paper,BCOR12mm,chapterprefix,liststotoc,bibtotoc,idxtotoc]{scrbook}
\LoadClass[dvips,10pt,a4paper,twoside,liststotoc,bibtotoc,idxtotoc,headinclude]{book}
\RequirePackage[a4paper,vdivide={*,22cm,4cm}]{geometry}
\fi


\RequirePackage{xcolor}
% jolis chapitres
\ifpdf
\RequirePackage[grey,utopia]{quotchap}
\else
\definecolor{chaptergrey}{rgb}{0.4,0.4,0.4}
\RequirePackage[nogrey,utopia]{quotchap}
\fi
\RequirePackage[Bjornstrup]{fncychap}

\RequirePackage{makeidx} % creation d'index
\RequirePackage[english,frenchb]{babel} % style francais
% pas d'espace automatiques avant les signes de ponctuation doubles !
\NoAutoSpaceBeforeFDP

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


\RequirePackage{amsmath,amssymb,amsthm,euscript}
% page de garde
\RequirePackage{pageGardeEnsieta}



\RequirePackage{tabularx}
\RequirePackage{eurosym} % pour le symbole euro
\RequirePackage{float}
\RequirePackage{multicol}

\RequirePackage{tipa}
\RequirePackage{lettrine}

\RequirePackage{pgf,tikz}
\usetikzlibrary{snakes,automata,shapes,plotmarks,chains,positioning,shadows,arrows}
\RequirePackage[explicit]{titlesec}
\newcommand*\chapterlabel{}
\titleformat{\chapter}
  {\gdef\chapterlabel{}
   \normalfont\sffamily\Huge\bfseries\scshape}
  {\gdef\chapterlabel{\thechapter\ }}{0pt}
  {\begin{tikzpicture}[remember picture,overlay]
    \node[yshift=-3cm] at (current page.north west)
      {\begin{tikzpicture}[remember picture, overlay]
        \draw[fill=blue!30] (0,0) rectangle
          (\paperwidth,3cm);
        \node[anchor=east,xshift=.9\paperwidth,rectangle,
              rounded corners=20pt,inner sep=11pt,
              fill=blue!50!black]
              {\color{white}\chapterlabel#1};
       \end{tikzpicture}
      };
   \end{tikzpicture}
  }
\titlespacing*{\chapter}{0pt}{50pt}{-60pt}
 


% modif des fontes pour le pdf
\ifpdf
%  \RequirePackage{mathptmx}
  \RequirePackage[scaled=.90]{helvet}
  \RequirePackage{courier}
\else
  % fontes lmodern
%  \RequirePackage{lmodern}
\fi
% aeguill correspond a ea plus les guillemets francais
\RequirePackage{aeguill}
\RequirePackage{yfonts}
\newcommand{\enluminure}[2]{\lettrine[lines=3]{\small \initfamily #1}{#2}}
%\RequirePackage{arabtex} % qq mots en arabe
\RequirePackage[latin1]{inputenc} % pour taper les accents directement
%\RequirePackage[annataritalic]{tengwarscript} % pour ecrire en elfique
% \frenchbsetup{AutoSpacePunctuation=false}  %frenchb recent
% paquetage ntheorem : permet de generer une table des theoremes
% attention : ntheorem a des bugs dans les algos de cesure (Overfull hbox)
%\RequirePackage[hyperref]{ntheorem}

%\RequirePackage{backref}

% hyperref doit etre charge en dernier
\ifpdf
\RequirePackage[pdftex,colorlinks]{hyperref}
\else
\RequirePackage[dvips]{hyperref}
\fi
%
\hypersetup{%
  pdftitle={Introduction à la recherche opérationnelle},
  pdfauthor={Rodéric Moitié},
  pdfkeywords={ro, recherche opérationnelle, graphes, simplexe, operations
    research, graphs, simplex},
  bookmarksnumbered,
  pdfstartview={FitH},
  citecolor=blue,
}%


\RequirePackage[french,undotted,tight]{minitoc}
\setcounter{tocdepth}{1}


\unitlength 1cm

% ensembles
\RequirePackage{dsfont}
\def\nbR{\mathds{R}}
\def\nbN{\mathds{N}}
\def\nbQ{\mathds{Q}}
\def\nbZ{\mathds{Z}}

\def\Coup{\ensuremath{\mathcal C}} 
\def\Cont{\ensuremath{\EuScript C}}
\def\Dom{\ensuremath{\EuScript D}}

\def\mcT{\ensuremath{\mathcal T}}
\def\mcF{\ensuremath{\mathcal F}}


\theoremstyle{plain}
\newtheorem{theo}{Théorème}
\newtheorem{lemme}{Lemme}
\newtheorem{corollaire}{Corollaire}
\theoremstyle{definition}
\newtheorem{defi}{Définition}
\theoremstyle{remark}
\newtheorem{exemple}{Exemple}
\newtheorem*{remarque}{Remarque}
\newtheorem*{remarques}{Remarques}
\newtheorem{prop}{Propriétés}

\newenvironment{preuve}[1] { {\bf Preuve
  } #1: \\} { \begin{flushright} $\blacksquare$ \end{flushright}}


\RequirePackage{listings}


\RequirePackage[toc,page]{appendix}
%%% cesures dans les references biblio avec apalike
% ne fonctionne qu'en pdf
\makeatletter
\let\@cite@ofmt\relax
\makeatother 

\makeatletter
% espace avant les ":" dans les figures
\renewcommand{\fnum@figure}{\figurename~\thefigure~}
\makeatother


\RequirePackage[algo2e,lined,algoruled,french]{algorithm2e}

\SetKwIF{If}{ElseIf}{Else}{si}{alors}{sinon si}{sinon}{fin si}
\SetKwFor{For}{pour}{faire}{fin pour}
\SetKwFor{While}{tant que}{faire}{fin tant que}
\SetKwFor{ForAll}{pour tous}{faire}{fin pour tous}
\SetKwRepeat{Repeat}{répéter}{jusqu'à}
\SetKwInput{KwData}{Données}
\SetKwInput{KwIn}{Entrées}

% \RequirePackage{algorithm}
% \floatname{algorithm}{Algorithme}
% \renewcommand{\listalgorithmname}{Liste des algorithmes}
% \RequirePackage{algpseudocode}
% \algblockdefx{Var}{EndVar}{\textbf{variables}}{\textbf{end variables}}
% \algblockdefx{Req}{EndReq}{\textbf{require :}}{\textbf{end require}}

% \renewcommand\algorithmicprocedure{\textbf{procédure}}
% \renewcommand\algorithmicfunction{\textbf{fonction}}
% \renewcommand\algorithmicreturn{\textbf{retourner}}
% \renewcommand\algorithmicfor{\textbf{pour}}
% \renewcommand\algorithmicforall{\textbf{pour tous}}
% \renewcommand\algorithmicdo{\textbf{faire}}
% \renewcommand\algorithmicwhile{\textbf{tant que}}
% \renewcommand\algorithmicrepeat{\textbf{répéter}}
% \renewcommand\algorithmicuntil{\textbf{jusqu'à}}
% \renewcommand\algorithmicif{\textbf{si}}
% \renewcommand\algorithmicthen{\textbf{alors}}
% \renewcommand\algorithmicelse{\textbf{sinon}}
% \renewcommand\algorithmicend{\textbf{fin}}
%\setlength{\algorithmicindent}{1cm}
%\algrenewcommand\algorithmicindent{1cm}


