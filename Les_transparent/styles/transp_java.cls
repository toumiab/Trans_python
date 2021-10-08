\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{transp_java}[2008/10/20]
\RequirePackage{ifpdf}


\ifpdf
\LoadClass[pdftex,beamer]{beamer}
\else
\LoadClass[xcolor=pst,dvips,handout]{beamer}
\fi
%\documentclass[handout]{beamer}


% Try the class options [notes], [notes=only], [trans], [handout],
% [red], [compress], [draft], [class=article] and see what happens!

% Copyright 2003 by Till Tantau <tantau@users.sourceforge.net>.
%
% This program can be redistributed and/or modified under the terms
% of the LaTeX Project Public License Distributed from CTAN
% archives in directory macros/latex/base/lppl.txt.


\RequirePackage{pgf,tikz}
\usetikzlibrary{snakes,plotmarks,positioning,shadows,arrows,shapes}
\RequirePackage{amsmath,amssymb,euscript}
\RequirePackage{eurosym}
\RequirePackage[latin1]{inputenc}
\RequirePackage{colortbl}
\RequirePackage[frenchb]{babel}
\RequirePackage{tipa}
% ajout malek 2016 pour la gestion de timer
%\RequirePackage[font=Times, timeinterval=10]{tickingclock}
\RequirePackage[font=Times,timeinterval=10, timeduration=4, timedeath=0, fillcolorwarningsecond=white!60!yellow,
timewarningfirst=50,timewarningsecond=80,resetatpages=2]{tdclock}
\newcommand{\chrono}{\hfill \toggleclock{\beamerbutton{toggle}}\color{red}\tdtime\resetcrono{\beamerbutton{reset}}}

%% version eleves
%\newcommand{\chrono}{}


\ifpdf
\graphicspath{{pdfs/}{imgs/}}
\DeclareGraphicsExtensions{.pdf,.png,.mps,.jpeg}
\else
\graphicspath{{epss/}}
% choix des extensions pour le mode dvips
%\DeclareGraphicsExtensions{.eps}
\fi

\RequirePackage{multicol}

\unitlength 1cm

% ensembles
\RequirePackage{dsfont}
\def\nbR{\mathds{R}}
\def\nbN{\mathds{N}}
\def\nbQ{\mathds{Q}}
\def\Coup{\ensuremath{\mathcal C}} 
\def\Dom{\ensuremath{\EuScript D}}
\def\Cont{\ensuremath{\EuScript C}}

\RequirePackage{listings}
\definecolor{green_comment}{rgb}{0.25,0.5,0.5}
\lstset{  basicstyle=\small\ttfamily\upshape,
          keywordstyle=\color{blue}\bfseries,
          identifierstyle=,
          commentstyle=\ttfamily\color{green_comment},
          stringstyle=\ttfamily\color{red},
          showstringspaces=false,
%          numbers=left,
%          firstnumber=1,
%          numberstyle=\tiny,
%          stepnumber=5,
          frame=single,
          language=Python}

\RequirePackage[algo2e,lined,algoruled,french]{algorithm2e}

\SetKwIF{If}{ElseIf}{Else}{si}{alors}{sinon si}{sinon}{fin si}
\SetKwFor{For}{pour}{faire}{fin pour}
\SetKwFor{While}{tant que}{faire}{fin tant que}
\SetKwFor{ForAll}{pour tous}{faire}{fin pour tous}
\SetKwInput{KwData}{Données}
\SetKwInput{KwIn}{Entrées}

% \RequirePackage{algorithm}
% \floatname{algorithm}{Algorithme}
% \renewcommand{\listalgorithmname}{Liste des algorithmes}
% \RequirePackage{algpseudocode}
% \algblockdefx{Var}{EndVar}{\textbf{variables}}{\textbf{fin variables}}
% \algblockdefx{Req}{EndReq}{\textbf{requiert :}}{\textbf{fin requiert}}
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
% %\setlength{\algorithmicindent}{1cm}
% \algrenewcommand\algorithmicindent{1cm}

\newcommand{\macoulr}[1]{\color{red}#1\color{black}}
\newcommand{\macoulb}[1]{\color{blue}#1\color{black}}
% Use some nice templates
%\beamertemplatetransparentcovereddynamic

\theoremstyle{definition}
\newtheorem{defi}{Définition}
\theoremstyle{example}
\newtheorem{remarque}{Remarque}
\newtheorem{remarques}{Remarques}
\newtheorem{exemple}{Exemple}
\newtheorem{exemples}{Exemples}
\theoremstyle{theorem}
\newtheorem{theoreme}{Théorème}
\newtheorem{propriete}{Propriétés}
\newtheorem{lemme}{Lemme}
\newtheorem{corollaire}{Corollaire}

