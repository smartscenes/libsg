#!/usr/bin/env bash

BACKGROUND="false"

# parse arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--background)
      BACKGROUND="true"
      shift # past value
      ;;
    -h|--help)
      echo "Usage:"
      echo "  -b, --background   run Flask server in background"
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ "$BACKGROUND" = true ]; then
    flask --app libsg.app run > /dev/null 2>&1 </dev/null &
else
    flask --app libsg.app --debug run
fi
